import requests
from bs4 import BeautifulSoup
import json
import re
import uuid
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse


class HRSDataScraper:
    def __init__(self, output_file="hrs_data_combined.json", url_log_file="processed_urls.json"):
        """
        Initialize the HRS Data Scraper

        :param output_file: Path to save combined JSON data
        :param url_log_file: Path to save processed URLs log
        """
        self.OUTPUT_FILE = output_file
        self.URL_LOG_FILE = url_log_file

    def extract_year_from_url(self, url):
        """Extract year from URL and return the wave label"""
        # Look for patterns like h04, h02, h06, etc. in the URL
        year_match = re.search(r'h(\d{2})lb', url)
        if year_match:
            # Convert 04 to 2004, 16 to 2016, etc.
            year_num = int(year_match.group(1))
            # Adjust for years before 2000 (e.g., 92 → 1992)
            full_year = year_num + (1900 if year_num >= 90 else 2000)
            return f"{full_year} Core"

        # Alternative pattern search
        year_match2 = re.search(r'/(19|20)(\d{2})/', url)
        if year_match2:
            return f"{year_match2.group(1)}{year_match2.group(2)} Core"

        # If no year found
        return "Unknown Core"

    def get_url_identifier(self, url):
        """Generate a consistent identifier for a URL to track processing"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        # Use the last part of the path (filename) if available
        if path:
            parts = path.split('/')
            return parts[-1].lower()
        return parsed.netloc.lower()

    def load_processed_urls(self):
        """Load previously processed URLs"""
        if os.path.exists(self.URL_LOG_FILE):
            try:
                with open(self.URL_LOG_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_processed_url(self, url, success=True, items_count=0):
        """Save a URL as processed with status"""
        processed_urls = self.load_processed_urls()
        url_id = self.get_url_identifier(url)
        processed_urls[url_id] = {
            "url": url,
            "success": success,
            "items_count": items_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.URL_LOG_FILE, 'w') as f:
            json.dump(processed_urls, f, indent=2)

    def is_valid_variable(self, item):
        """Check if the parsed item appears to be a valid variable entry."""
        if not item:
            return False

        # Must have a variable name that looks like a typical variable code (alphanumeric)
        if not item["variableName"] or not re.match(r'^[A-Z0-9]+$', item["variableName"]):
            return False

        # Should have either a description or responses
        if not item["description"] and not item["response"]:
            return False

        # Most valid variables have metadata
        if not item["Section"] and not item["Type"]:
            return False

        return True

    def clean_text(self,text):
        """Clean text by removing Unicode replacement characters and extra whitespace"""
        if not text:
            return ""
        # Remove Unicode replacement character
        text = text.replace('\ufffd', '')
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove newlines
        text = text.replace('\n', ' ')
        # Strip leading/trailing whitespace
        return text.strip()

    def parse_section(self, section_text, anchor_name=None):
        """Parse a section of text into structured variable data"""
        lines = section_text.strip().split('\n')

        # Skip sections that are too short
        if len(lines) < 2:
            return None

        item = {
            "variableName": "",
            "description": "",
            "Section": "",
            "Level": "",
            "Type": "",
            "Width": "",
            "Decimals": "",
            "CAI Reference": "",
            "question": "",
            "response": {}
        }

        # First set variableName from anchor if available
        if anchor_name and re.match(r'^[A-Z0-9]+$', anchor_name):
            item["variableName"] = anchor_name

        # Parse first non-empty line for variable name and description
        first_line_index = -1
        for i, line in enumerate(lines):
            if line.strip():
                first_line = line.strip()
                first_line_index = i

                # Try different patterns for variable name and description extraction
                match1 = re.match(r'^([A-Z0-9]+)\s{2,}(.+)$', first_line)
                match2 = re.match(r'^([A-Z0-9]+)\s+(.+)$', first_line)

                if match1:
                    item["variableName"] = match1.group(1).strip()
                    item["description"] = (match1.group(2))
                    break
                elif match2 and len(match2.group(1)) >= 3:  # Minimum length to avoid false positives
                    item["variableName"] = match2.group(1).strip()
                    item["description"] = self.clean_text(match2.group(2))
                    break
                break  # Only check the first non-empty line

        # Find metadata line
        metadata_line_index = -1
        for i, line in enumerate(lines):
            if 'Section:' in line and any(x in line for x in ['Level:', 'Type:', 'Width:']):
                metadata_line_index = i
                metadata_line = line.strip()

                # Extract metadata values
                section_match = re.search(r'Section:\s*(\w+)', metadata_line)
                level_match = re.search(r'Level:\s*(\w+)', metadata_line)
                type_match = re.search(r'Type:\s*(\w+)', metadata_line)
                width_match = re.search(r'Width:\s*(\d+)', metadata_line)
                decimals_match = re.search(r'Decimals:\s*(\d+)', metadata_line)

                if section_match: item["Section"] = section_match.group(1)
                if level_match: item["Level"] = level_match.group(1)
                if type_match: item["Type"] = type_match.group(1)
                if width_match: item["Width"] = width_match.group(1)
                if decimals_match: item["Decimals"] = decimals_match.group(1)
                break

        # Find CAI Reference
        cai_ref_index = -1
        for i, line in enumerate(lines):
            if 'CAI Reference:' in line:
                cai_parts = line.split('CAI Reference:', 1)
                if len(cai_parts) > 1:
                    item["CAI Reference"] = self.clean_text(cai_parts[1])
                    cai_ref_index = i
                    break
            elif re.search(r'CAI\s+Ref(erence)?:', line, re.IGNORECASE):
                cai_match = re.search(r'CAI\s+Ref(erence)?:\s*(.+)$', line, re.IGNORECASE)
                if cai_match:
                    item["CAI Reference"] = self.clean_text(cai_match.group(2))
                    cai_ref_index = i
                    break

        # Find separator line that marks beginning of responses
        response_start_index = -1
        for i, line in enumerate(lines):
            if i > metadata_line_index and metadata_line_index != -1:
                if (re.search(r'[-_.]{5,}', line) or
                        re.search(r'\.\s*\.\s*\.', line) or
                        re.search(r'\.{5,}', line) or
                        line.strip() == '-' * len(line.strip()) or
                        '______' in line or
                        all(c in '-_.' for c in line.strip())):
                    response_start_index = i
                    break

        # If no separator found, look for response-like patterns
        if response_start_index == -1 and metadata_line_index != -1:
            for i, line in enumerate(lines):
                if i > metadata_line_index:
                    if (re.match(r'^\s*\d+\s+\d+\.', line) or
                            re.match(r'^\s*\d+\.\s+.+\d+$', line) or
                            re.match(r'^\s*\d+\s+[A-Za-z]', line) or
                            re.match(r'^\s*[A-Za-z0-9.-]+\s+\d+$', line) or
                            re.match(r'^\s*\d+\.\s+.+', line) and i > metadata_line_index + 3):
                        response_start_index = i - 1
                        break

        # Extract question text - starting after metadata
        if metadata_line_index != -1:
            question_start_index = metadata_line_index + 1

            # Find where question text ends
            question_end_index = -1
            for i in range(question_start_index, len(lines)):
                line = lines[i].strip()
                if re.search(r'\.{5,}', line) or re.search(r'\.\s*\.\s*\.', line):
                    question_end_index = i
                    break

            # If no end marker found but responses start is known, use that
            if question_end_index == -1 and response_start_index != -1:
                question_end_index = response_start_index

            # If we found a valid range for question text
            if question_end_index > question_start_index:
                # Skip the CAI Reference line if present
                question_lines = []
                for i in range(question_start_index, question_end_index):
                    line = lines[i].strip()
                    # Skip empty lines at beginning
                    if not line and not question_lines:
                        continue
                    # Skip lines with CAI Reference
                    if 'CAI Reference:' in line or re.search(r'CAI\s+Ref(erence)?:', line, re.IGNORECASE):
                        continue
                    question_lines.append(line)

                # Combine lines and clean up
                if question_lines:
                    question_text = '\n'.join(question_lines)
                    # Remove "Ask:" prefix if present
                    question_text = re.sub(r'^Ask:\s*', '', question_text)
                    item["question"] = self.clean_text(question_text)

        # Extract responses
        if response_start_index != -1:
            # Check for checkbox/multiple choice patterns
            is_checkbox_list = False
            num_options_without_counts = 0

            for i in range(response_start_index + 1, min(response_start_index + 10, len(lines))):
                line = lines[i].strip()
                if not line:
                    continue
                if re.match(r'^\d+\.\s+.+$', line) and not re.match(r'^\d+\.\s+.+\d+$', line):
                    num_options_without_counts += 1

            if num_options_without_counts >= 2:
                is_checkbox_list = True

            # Process lines after response separator
            numeric_values_found = False
            processed_stats_line = False
            current_description = []
            current_count = None
            current_number = None

            for i in range(response_start_index + 1, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                # Pattern for descriptive statistics with multiple columns
                if line.strip() == "-----------------------------------------------------------------":
                    continue

                # Check if this is a statistics header/value line
                stats_match = re.match(
                    r'^\s*(N|Min|Max|Mean|SD|Miss)\s+(N|Min|Max|Mean|SD|Miss)\s+(N|Min|Max|Mean|SD|Miss)\s+(N|Min|Max|Mean|SD|Miss)\s+(N|Min|Max|Mean|SD|Miss)\s+(N|Min|Max|Mean|SD|Miss)\s*$',
                    line)

                if stats_match and i + 1 < len(lines):
                    value_line = lines[i + 1].strip()
                    values = re.findall(r'(\d+(?:\.\d+)?)', value_line)
                    headers = ['N', 'Min', 'Max', 'Mean', 'SD', 'Miss']
                    if len(values) == len(headers):
                        for header, value in zip(headers, values):
                            try:
                                if '.' in value:
                                    item["response"][header] = float(value)
                                else:
                                    item["response"][header] = int(value)
                            except ValueError:
                                item["response"][header] = value
                        processed_stats_line = True
                        continue
                if processed_stats_line:
                    processed_stats_line = False
                    continue

                # Pattern 1: Number at beginning followed by option and text (for multi-line descriptions)
                match_multi_start = re.match(r'^(\d+)\s+(\d+\.)\s+(.+)$', line)

                if match_multi_start:
                    # Save previous description if exists
                    if current_description and current_count is not None and current_number is not None:
                        full_description = self.clean_text(" ".join(current_description))
                        label = f"{current_number} {full_description}"
                        item["response"][label] = current_count

                    # Start new response item
                    current_count = int(match_multi_start.group(1))
                    current_number = match_multi_start.group(2)
                    current_description = [match_multi_start.group(3)]
                    continue

                # Check if this is continuation of description
                elif current_description and (line.startswith(' ') or re.match(r'^[a-zA-Z]', line)):
                    current_description.append(line.strip())
                    continue

                # If we're here, save current multi-line item if exists
                if current_description and current_count is not None and current_number is not None:
                    full_description = self.clean_text(" ".join(current_description))
                    label = f"{current_number} {full_description}"
                    item["response"][label] = current_count
                    current_description = []
                    current_count = None
                    current_number = None

                # Now try other patterns from script 1
                # Pattern 2: Option number/letter followed by text and number at end
                match1 = re.match(r'^(\d+\.|\w\.)\s+(.+?)\s+(\d+)$', line)

                # Pattern 3: Text followed by number at end
                match2 = re.match(r'^(.+?)\s+(\d+)$', line)

                # Pattern 4: Number at beginning followed by option and text (single line)
                match3 = re.match(r'^(\d+)\s+(\d+\.|\w\.)\s+(.+)$', line)

                # Pattern 5: Number at beginning followed by text
                match4 = re.match(r'^(\d+)\s+(.+)$', line)

                # Pattern 6: Just numbered options
                match5 = re.match(r'^(\d+\.|\w\.)\s+(.+)$', line)

                # Track if we found any numeric values for responses
                if match1 or match2 or match3 or match4:
                    numeric_values_found = True

                if match1:
                    option = match1.group(1).strip()
                    text = self.clean_text(match1.group(2))
                    count = int(match1.group(3))
                    label = f"{option} {text}"
                    if not re.search(r'Section:\s+\w+', label):
                        item["response"][label] = count
                elif match2:
                    label = self.clean_text(match2.group(1))
                    count = int(match2.group(2))
                    if not re.search(r'Section:\s+\w+', label):
                        item["response"][label] = count
                elif match3 and not match_multi_start:  # Avoid double-matching
                    count = int(match3.group(1))
                    option = match3.group(2).strip()
                    text = self.clean_text(match3.group(3))
                    label = f"{option} {text}"
                    if not re.search(r'Section:\s+\w+', label):
                        item["response"][label] = count
                elif match4:
                    # More careful matching for pattern 4 to avoid false positives
                    count_str = match4.group(1)
                    remaining_text = match4.group(2).strip()

                    # Skip if this looks like a pattern 1 case (has option marker immediately after count)
                    if re.match(r'^(\d+\.|\w\.)\s+', remaining_text):
                        continue

                    # Skip if the label contains common metadata patterns
                    if any(keyword in remaining_text for keyword in
                           ['Section:', 'Level:', 'Type:', 'Width:', 'Decimals:', 'CAI Reference', 'Person Identifier',
                            'Household ID']):
                        continue

                    # Only process if the count is a valid integer and label looks like a response option
                    try:
                        count = int(count_str)
                        if count > 0 and count < 100000:  # Reasonable range for count values
                            item["response"][self.clean_text(remaining_text)] = count
                    except ValueError:
                        continue
                elif match5 and (is_checkbox_list or not numeric_values_found):
                    option = match5.group(1).strip()
                    text = self.clean_text(match5.group(2))
                    label = f"{option} {text}"
                    if not re.search(r'Section:\s+\w+', label):
                        item["response"][label] = ""  # Use empty string instead of None
                else:
                    # Check for special entries like "Blank. INAP"
                    if "blank" in line.lower() or "inap" in line.lower() or "inapplicable" in line.lower():
                        number_match = re.search(r'(\d+)$', line)
                        if number_match:
                            count = int(number_match.group(1))
                            label_parts = re.split(r'\s+' + str(count) + r'$', line)
                            if label_parts:
                                label = self.clean_text(label_parts[0])
                                item["response"][label] = count
                        else:
                            label = self.clean_text(line)
                            item["response"][label] = ""  # Use empty string instead of None

            # Don't forget to save the last multi-line item
            if current_description and current_count is not None and current_number is not None:
                full_description = self.clean_text(" ".join(current_description))
                label = f"{current_number} {full_description}"
                item["response"][label] = current_count

        # Final validation for variable name
        if item["variableName"] == "" and anchor_name and re.match(r'^[A-Z0-9]+$', anchor_name):
            item["variableName"] = anchor_name

            # Try to extract description from first line if it exists and wasn't already set
            if item["description"] == "" and first_line_index != -1:
                first_line = lines[first_line_index].strip()
                if anchor_name in first_line:
                    description_part = first_line.split(anchor_name, 1)[1].strip()
                    if description_part:
                        item["description"] = self.clean_text(description_part)

        # Clean up question text
        if item["question"]:
            # Remove leading "Ask:" if present
            item["question"] = re.sub(r'^Ask\s*:', '', item["question"]).strip()
            # Clean up any remaining special patterns or markers
            item["question"] = re.sub(r'^IF\s+\(.*?\)\s*', '', item["question"]).strip()
            # Final cleaning
            item["question"] = self.clean_text(item["question"])

        return item

    def scrape_hrs_data(self, url, verbose=True):
        """Scrape data from a single HRS URL"""
        if verbose:
            print(f"Processing URL: {url}")

        # Check if already processed
        processed_urls = self.load_processed_urls()
        url_id = self.get_url_identifier(url)

        if url_id in processed_urls and processed_urls[url_id]["success"]:
            if verbose:
                print(f"Skipping already processed URL: {url}")
            return None

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': url
        }

        try:
            # Add rate limiting to avoid hammering the server
            time.sleep(1)

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the correct frame containing the data
            r_frame_url = next((frame['src'] for frame in soup.find_all('frame') if '_r.htm' in frame.get('src', '')),
                               None)
            if not r_frame_url:
                if verbose:
                    print(f"Error: Could not find frame with '_r.htm' in the src attribute for {url}")
                self.save_processed_url(url, success=False)
                return None

            base_url = url.rsplit('/', 1)[0]
            absolute_r_frame_url = f"{base_url}/{r_frame_url.strip()}"

            frame_response = requests.get(absolute_r_frame_url, headers=headers, timeout=30)
            frame_response.raise_for_status()

            # Parse the frame content as HTML
            frame_soup = BeautifulSoup(frame_response.content, 'html.parser')
            raw_html = str(frame_soup)

            # Extract the wave year from URL
            wave = self.extract_year_from_url(url)

            # Split by anchor tags which typically mark the start of variable sections
            sections = re.split(r'<a name="([^"]+)"></a>', raw_html)

            # Process sections (skip first element, then process pairs)
            data = []
            for i in range(1, len(sections) - 1, 2):
                anchor_name = sections[i]
                content = sections[i + 1]

                # Skip sections with known non-variable anchors
                if anchor_name.lower() in ['top', 'bottom', 'toc', 'index', 'contents']:
                    continue

                # Create a soup object for this section's content
                section_soup = BeautifulSoup(content, 'html.parser')
                section_text = section_soup.get_text(separator='\n').strip()

                # Parse the section
                item = self.parse_section(section_text, anchor_name)

                # Check if this is a valid variable item
                if self.is_valid_variable(item):
                    # Add UUID and wave to the item
                    item["id"] = str(uuid.uuid4())
                    item["wave"] = wave
                    data.append(item)

            # Save as processed
            self.save_processed_url(url, success=True, items_count=len(data))

            if verbose:
                print(f"Successfully extracted {len(data)} items from {url}")
            return data

        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"Request Error for {url}: {e}")
            self.save_processed_url(url, success=False)
            return None
        except Exception as e:
            if verbose:
                print(f"An error occurred for {url}: {e}")
            self.save_processed_url(url, success=False)
            return None

    def process_urls_from_file(self, url_file_path, max_workers=3, verbose=True):
        """
        Process multiple URLs from a file with threading

        :param url_file_path: Path to file containing URLs to process
        :param max_workers: Number of concurrent threads to use
        :param verbose: Whether to print progress information
        :return: List of all extracted data items
        """
        # Load URLs from file
        with open(url_file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if verbose:
            print(f"Loaded {len(urls)} URLs from {url_file_path}")

        # Initialize or load existing combined data
        all_data = []
        if os.path.exists(self.OUTPUT_FILE):
            try:
                with open(self.OUTPUT_FILE, 'r') as f:
                    all_data = json.load(f)
                if verbose:
                    print(f"Loaded {len(all_data)} existing items from {self.OUTPUT_FILE}")
            except json.JSONDecodeError:
                if verbose:
                    print(f"Error loading {self.OUTPUT_FILE}, starting with empty dataset")

        # Process URLs with thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda url: self.scrape_hrs_data(url, verbose), urls))

        # Combine results
        new_items_count = 0
        for result in results:
            if result:
                all_data.extend(result)
                new_items_count += len(result)

        # Save combined data
        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(all_data, f, indent=2)

        if verbose:
            print(f"Batch processing complete. Added {new_items_count} new items. Total: {len(all_data)} items.")

        return all_data

    def process_single_url(self, url):
        """
        Process a single URL and save to output file

        :param url: URL to process
        :return: List of extracted data items
        """
        # Process single URL
        data = self.scrape_hrs_data(url)

        # Save to combined file
        all_data = []
        if os.path.exists(self.OUTPUT_FILE):
            try:
                with open(self.OUTPUT_FILE, 'r') as f:
                    all_data = json.load(f)
            except:
                pass

        if data:
            all_data.extend(data)

        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(all_data, f, indent=2)

        print(f"Single URL processing complete. Total items: {len(all_data)}")
        return all_data


# Example usage
if __name__ == "__main__":
    # Example of processing URLs from a file
    scraper = HRSDataScraper()
    scraper.process_urls_from_file('urls.txt')

    # Example of processing a single URL
    # scraper.process_single_url('https://example.com/hrs_data_url')