import csv
import webbrowser
import time

def search_links(csv_file, num_rows):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        link_count = 0
        for row in reader:
            if link_count == num_rows:
                break
            link = row['url']
            if link.startswith('http://') or link.startswith('https://'):
                webbrowser.open(link)
                link_count += 1

# CSV file path
csv_file_name = '/home/josva/Pictures/yolo-ecommerce-app/results.csv'

# Number of rows to search at a time
rows_per_set = 3

with open(csv_file_name, 'r') as file:
    reader = csv.DictReader(file)
    total_rows = sum(1 for _ in reader)

current_row = 0
while current_row < total_rows:
    print("Searching links...")
    search_links(csv_file_name, rows_per_set)
    current_row += rows_per_set

    # Wait for 20 seconds
    print("Waiting for 20 seconds...")
    time.sleep(20)

    # Ask the user to continue
    user_input = input("Continue with the next set of links? (y/n): ")
    if user_input.lower() != 'y':
        break

