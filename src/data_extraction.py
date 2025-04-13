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

    def parse_section(self, section_text, anchor_name=None):
        """Parse a section of text into structured variable data"""
        # The original parse_section method remains unchanged
        # (Paste the entire parse_section method from the original script here)
        # ... (keep the exact same implementation as in the original script)

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