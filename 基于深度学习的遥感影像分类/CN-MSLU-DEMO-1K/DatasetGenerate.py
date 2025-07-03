import os
import xml.etree.ElementTree as ET
import pandas as pd
import concurrent.futures



def process_xml_file(xml_file):

    # Parsing XML files
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the required attributes
    folder = root.find('folder').text
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    firstlevel = root.find('category/firstlevel').text
    secondlevel = root.find('category/secondlevel').text
    filepath = os.path.join(TIF_DIRECTORY,folder,filename)
    print(folder, filepath, os.path.exists(filepath))
    assert os.path.exists(filepath)
    return [folder, filename, filepath, width, height, firstlevel, secondlevel]



def generate(directory):
    # Define a list of data to generate a dataframe
    data = []

    # for filename in os.listdir(directory):
    #     if not filename.endswith('.xml'):
    #         continue
    #     xml_path = os.path.join(directory, filename)
    #
    #     data.append(process_xml_file(xml_path))


    # Traverse all files in the file directory
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for filename in os.listdir(directory):
            if not filename.endswith('.xml'):
                continue
            xml_path = os.path.join(directory, filename)
            future = executor.submit(process_xml_file, xml_path)
            data.append(future.result())


    df = pd.DataFrame(data, columns=['Class', 'Filename', 'Filepath', 'Width', 'Height', 'FirstLevel', 'SecondLevel'])
    df.to_csv("CN-MSLU-DEMO-1K.csv",index=None)


if __name__ == '__main__':
    # TODO: Change to your start `directory` and `max_workers` here
    DIRECTORY = r''
    NUM_WORKERS = 4


    XML_DIRECTORY = os.path.join(DIRECTORY, "Classification")
    TIF_DIRECTORY = os.path.join(DIRECTORY, "ImageSets")
    generate(XML_DIRECTORY)