import cv2
import numpy as np
import logging

#set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def image_data_extraction(path: str) -> np.ndarray:
    # extracts HSV data from an image
    try:
        image = cv2.cvtColor(cv2.imread(path) ,cv2.COLOR_BGR2HSV_FULL)
        data = np.dstack((image[..., 0], image[..., 1], image[..., 2]))   
        return data.astype(np.uint8)
    except Exception as e:
        logging.error(f"\n\nError reading or processing image at {path}:\n   {e}\n")
        raise

def recurring_identify(data: np.ndarray, data_channel: int) -> list:
    # identifies and sorts recurring HSV values in a specified channel
    try:
        channel_counts = np.bincount(data[:, :, data_channel].flatten())
        non_zero_indices = np.nonzero(channel_counts)[0]
        sorted_indices = non_zero_indices[channel_counts[non_zero_indices].argsort()[::-1]]
        return list(sorted_indices)
    except Exception as e:
        logging.error(f"\n\nError identifying recurring values:\n   {e}\n")
        raise

def dictionary_mapping(source_arr: list, target_arr: list) -> dict:
    # maps values from source to target based on frequency
    try: 
        mapped_dict = {}
        if (len(source_arr) >= len(target_arr)):
            for i in range(0,len(target_arr)):
                mapped_dict[target_arr[i]] = source_arr[i]
            return mapped_dict
        else:
            for i in range(0,len(source_arr)):
                mapped_dict[target_arr[i]] = source_arr[i]
            remaining = len(target_arr) - (len(target_arr) - len(source_arr))
            for i in range(len(target_arr) - (len(target_arr) - len(source_arr)), len(target_arr)):
                if (i - remaining == len(target_arr) - (len(target_arr) - len(source_arr))):
                    remaining = remaining + len(target_arr) - (len(target_arr) - len(source_arr))
                mapped_dict[target_arr[i]] = source_arr[i - remaining]
        return mapped_dict
    except Exception as e:
        logging.error(f"\n\nError mapping dictionary:\n   {e}\n")
        raise

def apply_mapping(channel_data: np.ndarray, mapped_dictionary: dict) -> np.ndarray:
    # applies mapping to channel data
    try:
        mask = np.isin(channel_data, list(mapped_dictionary.keys()))
        vectorized_map = np.vectorize(mapped_dictionary.get)
        channel_data[mask] = vectorized_map(channel_data[mask])
        return channel_data
    except Exception as e:
        logging.error(f"\n\nError applying mapping:\n   {e}\n")

def swap_channel_data(data: np.ndarray, data_channel: int, mapped_dictionary: dict) -> np.ndarray:
    # swaps channel data based on mapping
    try:
        data = np.array(data)
        channel_data = data[:, :, data_channel]
        channel_data = apply_mapping(channel_data, mapped_dictionary)
        data[:, :, data_channel] = channel_data
        return data
    except Exception as e:
        logging.error(f"\n\nError swapping channel data:\n   {e}\n")
        raise
    

def image_data_swap(target_data: np.ndarray, mapped_dictionary: dict, data_channel: int) -> np.ndarray:
    # main function to swap data in a specified channel
    return swap_channel_data(target_data, data_channel, mapped_dictionary)


def process_images(source: str, target: str, data_channel: int) -> None:
    try:
        # extract data from the source and target images
        source_data = image_data_extraction(source)
        target_data = image_data_extraction(target)

        # identify the recurring HSV values in the specified data channel
        source_array = recurring_identify(source_data, data_channel)
        target_array = recurring_identify(target_data, data_channel)

        # map the most recurring HSV values between source and target
        mapped_dict = dictionary_mapping(source_array, target_array)

        # swap the data in the specified channel using the mapped dictionary
        swapped_data = image_data_swap(target_data, mapped_dict, data_channel)

        # Convert swapped data back to BGR color space for saving/viewing
        swapped_image = cv2.cvtColor(swapped_data, cv2.COLOR_HSV2BGR_FULL)
        
        # Save the swapped image
        output_path = "output_image.png"
        cv2.imwrite(output_path, swapped_image)

        logging.info(f"\n\nNew image saved as {output_path}\n")
        
    except Exception as e:
        logging.error(f"\n\nAn error occurred in main:\n   {e}\n")