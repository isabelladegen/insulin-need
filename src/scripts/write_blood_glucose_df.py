from src.configurations import Configuration
from src.read import read_all_bg
from src.write import write_read_record

flat_file_folder = '../../data'
per_id_folder = flat_file_folder + '/perid'


# set as_flat_file to True if you want to save one big flat bg csv or set it to false if you want a bg_csv per id
def main():
    as_flat_file = False
    folder = flat_file_folder if as_flat_file else per_id_folder
    config = Configuration()
    result = read_all_bg(config)
    write_read_record(result, as_flat_file, folder, config.bg_file)


if __name__ == "__main__":
    main()
