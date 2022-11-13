from src.configurations import Configuration
from src.read import read_all_device_status
from src.write import write_read_record


def main():
    config = Configuration()
    as_flat_file = config.as_flat_file
    folder = '../' + config.data_folder if as_flat_file else '../' + config.perid_data_folder
    result = read_all_device_status(config)
    write_read_record(result, as_flat_file, folder, config.flat_device_status_116_file_name)


if __name__ == "__main__":
    main()
