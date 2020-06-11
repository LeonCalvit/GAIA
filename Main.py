import csv
import requests
import gzip
import io
from bs4 import BeautifulSoup


def import_data(base_url: str, url: str, file_writer, number=0):
    url = str(base_url) + url
    # Get .gz file from URL
    r = requests.get(url, allow_redirects=True)
    # Decompress .gz in memory
    compressed_file = io.BytesIO(r.content)
    decompressed_file = gzip.open(compressed_file, mode='rt')
    csv_table = csv.reader(decompressed_file)
    print('Exporting file ', number)

    # Skip first row as it's just labels
    iter_row = iter(csv_table)
    next(iter_row)
    for row in csv_table:
        data = []
        data.append(row[1])  # Designation
        data.append(row[5])  # right ascension
        data.append(row[7])  # declination
        data.append(row[3])  # Random Index
        data.append(row[9])  # Parallax
        data.append(row[78])  # Temperature
        data.append(row[91])  # Luminosity
        data.append(row[66])  # Radial Velocity
        data.append(row[88])  # radius
        file_writer.writerow(data)
        del data

    print('Finished file ', number)


def create_csv(file_name, size):
    # Get all links on the appropriate page for downloading the data
    base_link = '''http://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'''
    f = requests.get(base_link)
    page = f.text
    soup = BeautifulSoup(page, "html.parser").findAll('a')
    # The first link is just a thing to go back to the previous page
    del soup[0]
    with open(file_name, 'w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        # loops through links, will call import data on each link
        x = 1
        for link in soup:
            import_data(base_link, link.get('href'), file_writer, number=x)
            x += 1
            # Limit number of files to download for testing
            # print(f'{x}/400')
            if x > size:
                break

    print('All done')


if __name__ == "__main__":
    input("This download will take a long time.  Press anything to continue.")
    create_csv('GAIA_DATA.csv', 400)
    create_csv('GAIA_DATA_small.csv', 1)
