from auxiliarymethods import dataset_parsers as dp

def main():
    dataset = "short-slize-8-12"
    graph_db, classes = dp.read_txt(dataset)


if __name__ == "__main__":
    main()
