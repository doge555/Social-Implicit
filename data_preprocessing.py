import numpy as np

def loading_data(input_file=str, output_file=str):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    data = []
    with open(output_file, 'w') as file:
        for line in lines:
            values = line.strip().split(',')
            frame = float(values[0])
            id = float(values[1])
            bbox_center_x = (float(values[2]) + float(values[4]))/2
            bbox_center_y = (float(values[3]) + float(values[5]))/2
            data.append([frame, id, bbox_center_x, bbox_center_y])
    data_array = np.array(data)
    data_array_after_pruning = preprocessing_data(data_array=data_array, threshold=20)
    np.savetxt(output_file, data_array_after_pruning, delimiter='\t')
    return 1

def preprocessing_data(data_array=np.array, threshold=int):
    data_array_after_pruning = data_array
    frames = np.unique(data_array_after_pruning[:, 0])
    id_temp_map = None
    current_frame_idx = 0
    last_losted_id_info = None
    for frame in frames:
        if id_temp_map is None:
            ids_in_begining_frame = data_array_after_pruning[data_array_after_pruning[:, 0] == frame, 1]
            current_frame_idx += len(ids_in_begining_frame)
            id_temp_map = dict(zip(map(float, ids_in_begining_frame), map(int, np.zeros_like(ids_in_begining_frame))))
            last_losted_id_info = dict(zip(map(float, ids_in_begining_frame), data_array_after_pruning[data_array_after_pruning[:, 0] == frame, :]))
        else:
            ids_in_curr_frame = data_array_after_pruning[data_array_after_pruning[:, 0] == frame, 1]
            current_frame_idx += len(ids_in_curr_frame)
            # add new id in current frame into dict which saved id for last frame.
            for id in ids_in_curr_frame:
                if not np.isin(id, list(id_temp_map.keys())):
                    id_temp_map[id] = 0
            # check id in last frame but not in current frame and increment the losing time by 1.
            for id in id_temp_map:
                if not np.isin(id, ids_in_curr_frame):
                    id_temp_map[id] += 1
                else:
                    if id_temp_map[id] != 0:
                        previous_frame_idx = current_frame_idx
                        correspond_frame_array = data_array_after_pruning[data_array_after_pruning[:, 0] == frame, :]
                        correspond_id_array = correspond_frame_array[correspond_frame_array[:, 1] == id][0]
                        center_x_diff = abs(last_losted_id_info[id][2] - correspond_id_array[2])/(id_temp_map[id] + 1)
                        center_y_diff = abs(last_losted_id_info[id][3] - correspond_id_array[3])/(id_temp_map[id] + 1)
                        for i in range(id_temp_map[id]):
                            previous_frame_idx -= len(data_array_after_pruning[data_array_after_pruning[:, 0] == frame-i, :])
                            if correspond_id_array[2] > last_losted_id_info[id][2]:
                                dummy_center_x =  correspond_id_array[2] - center_x_diff
                            else:
                                dummy_center_x =  correspond_id_array[2] + center_x_diff
                            if correspond_id_array[3] > last_losted_id_info[id][3]:
                                dummy_center_y =  correspond_id_array[3] - center_y_diff
                            else:
                                dummy_center_y =  correspond_id_array[3] + center_y_diff
                            dummy_row = [frame-i-1, id, dummy_center_x, dummy_center_y]
                            data_array_after_pruning = np.insert(data_array_after_pruning, previous_frame_idx, dummy_row, axis=0)
                            previous_frame_idx += 1
                            current_frame_idx += 1
                        selected_array = data_array_after_pruning[data_array_after_pruning[:, 0] == frame, :]
                        last_losted_id_info[id] = selected_array[selected_array[:, 1] == id][0]
                        id_temp_map[id] = 0
                    else:
                        selected_array = data_array_after_pruning[data_array_after_pruning[:, 0] == frame, :]
                        last_losted_id_info[id] = selected_array[selected_array[:, 1] == id][0]
            # delete rows up to the current frame with id which lost for 20 losing time long
            for id in list(id_temp_map.keys()):
                if id_temp_map[id] > threshold:
                    mask = data_array_after_pruning[:current_frame_idx, 1] != id
                    truncated_data_array = data_array_after_pruning[:current_frame_idx][mask]
                    rest_data_array = data_array_after_pruning[current_frame_idx:]
                    current_frame_idx = len(truncated_data_array)
                    data_array_after_pruning = np.concatenate((truncated_data_array, rest_data_array), axis=0)
                    del id_temp_map[id]
                    del last_losted_id_info[id]
    return data_array_after_pruning

            

            

if __name__ == "__main__":
    # train
    input_file_1 = "../AI_city_dataset/S002/c008/label.txt"
    output_file_1 = "../AI_city_dataset/AICity_after_precessing/train/market_1_train.txt"
    
    input_file_2 = "../AI_city_dataset/S002/c009/label.txt"
    output_file_2 = "../AI_city_dataset/AICity_after_precessing/train/market_2_train.txt"
    
    input_file_3 = "../AI_city_dataset/S002/c010/label.txt"
    output_file_3 = "../AI_city_dataset/AICity_after_precessing/train/market_3_train.txt"
    
    input_file_4 = "../AI_city_dataset/S002/c011/label.txt"
    output_file_4 = "../AI_city_dataset/AICity_after_precessing/train/market_4_train.txt"

    input_file_5 = "../AI_city_dataset/S004/c020/label.txt"
    output_file_5 = "../AI_city_dataset/AICity_after_precessing/train/market_5_train.txt"
    
    input_file_6 = "../AI_city_dataset/S004/c021/label.txt"
    output_file_6 = "../AI_city_dataset/AICity_after_precessing/train/market_6_train.txt"
    
    input_file_7 = "../AI_city_dataset/S004/c022/label.txt"
    output_file_7 = "../AI_city_dataset/AICity_after_precessing/train/market_7_train.txt"
    
    input_file_8 = "../AI_city_dataset/S004/c024/label.txt"
    output_file_8 = "../AI_city_dataset/AICity_after_precessing/train/market_8_train.txt"

    input_file_9 = "../AI_city_dataset/S007/c036/label.txt"
    output_file_9 = "../AI_city_dataset/AICity_after_precessing/train/hallway_1_train.txt"
    
    input_file_10 = "../AI_city_dataset/S007/c037/label.txt"
    output_file_10 = "../AI_city_dataset/AICity_after_precessing/train/hallway_2_train.txt"
    
    input_file_11 = "../AI_city_dataset/S007/c038/label.txt"
    output_file_11 = "../AI_city_dataset/AICity_after_precessing/train/hallway_3_train.txt"
    
    input_file_12 = "../AI_city_dataset/S007/c039/label.txt"
    output_file_12 = "../AI_city_dataset/AICity_after_precessing/train/hallway_4_train.txt"

    input_file_13 = "../AI_city_dataset/S010/c053/label.txt"
    output_file_13 = "../AI_city_dataset/AICity_after_precessing/train/hallway_5_train.txt"
    
    input_file_14 = "../AI_city_dataset/S010/c054/label.txt"
    output_file_14 = "../AI_city_dataset/AICity_after_precessing/train/hallway_6_train.txt"
    
    input_file_15 = "../AI_city_dataset/S010/c055/label.txt"
    output_file_15 = "../AI_city_dataset/AICity_after_precessing/train/hallway_7_train.txt"
    
    input_file_16 = "../AI_city_dataset/S010/c056/label.txt"
    output_file_16 = "../AI_city_dataset/AICity_after_precessing/train/hallway_8_train.txt"

    input_file_17 = "../AI_city_dataset/S011/c059/label.txt"
    output_file_17 = "../AI_city_dataset/AICity_after_precessing/train/hospital_1_train.txt"
    
    input_file_18 = "../AI_city_dataset/S011/c060/label.txt"
    output_file_18 = "../AI_city_dataset/AICity_after_precessing/train/hospital_2_train.txt"
    
    input_file_19 = "../AI_city_dataset/S011/c061/label.txt"
    output_file_19 = "../AI_city_dataset/AICity_after_precessing/train/hospital_3_train.txt"
    
    input_file_20 = "../AI_city_dataset/S011/c062/label.txt"
    output_file_20 = "../AI_city_dataset/AICity_after_precessing/train/hospital_4_train.txt"

    input_file_21 = "../AI_city_dataset/S012/c065/label.txt"
    output_file_21 = "../AI_city_dataset/AICity_after_precessing/train/hospital_5_train.txt"
    
    input_file_22 = "../AI_city_dataset/S012/c066/label.txt"
    output_file_22 = "../AI_city_dataset/AICity_after_precessing/train/hospital_6_train.txt"
    
    input_file_23 = "../AI_city_dataset/S012/c067/label.txt"
    output_file_23 = "../AI_city_dataset/AICity_after_precessing/train/hospital_7_train.txt"
    
    input_file_24 = "../AI_city_dataset/S012/c068/label.txt"
    output_file_24 = "../AI_city_dataset/AICity_after_precessing/train/hospital_8_train.txt"

    input_file_25 = "../AI_city_dataset/S015/c082/label.txt"
    output_file_25 = "../AI_city_dataset/AICity_after_precessing/train/factory_1_train.txt"
    
    input_file_26 = "../AI_city_dataset/S015/c083/label.txt"
    output_file_26 = "../AI_city_dataset/AICity_after_precessing/train/factory_2_train.txt"
    
    input_file_27 = "../AI_city_dataset/S015/c084/label.txt"
    output_file_27 = "../AI_city_dataset/AICity_after_precessing/train/factory_3_train.txt"
    
    input_file_28 = "../AI_city_dataset/S015/c085/label.txt"
    output_file_28 = "../AI_city_dataset/AICity_after_precessing/train/factory_4_train.txt"
    
    input_file_29 = "../AI_city_dataset/S016/c088/label.txt"
    output_file_29 = "../AI_city_dataset/AICity_after_precessing/train/factory_5_train.txt"
    
    input_file_30 = "../AI_city_dataset/S016/c089/label.txt"
    output_file_30 = "../AI_city_dataset/AICity_after_precessing/train/factory_6_train.txt"
    
    input_file_31 = "../AI_city_dataset/S016/c090/label.txt"
    output_file_31 = "../AI_city_dataset/AICity_after_precessing/train/factory_7_train.txt"
    
    input_file_32 = "../AI_city_dataset/S016/c091/label.txt"
    output_file_32 = "../AI_city_dataset/AICity_after_precessing/train/factory_8_train.txt"

    # validation
    input_file_val_1 = "../AI_city_dataset/S005/c025/label.txt"
    output_file_val_1 = "../AI_city_dataset/AICity_after_precessing/validation/market_1_val.txt"

    input_file_val_2 = "../AI_city_dataset/S005/c026/label.txt"
    output_file_val_2 = "../AI_city_dataset/AICity_after_precessing/validation/market_2_val.txt"

    input_file_val_3 = "../AI_city_dataset/S005/c027/label.txt"
    output_file_val_3 = "../AI_city_dataset/AICity_after_precessing/validation/market_3_val.txt"

    input_file_val_4 = "../AI_city_dataset/S005/c028/label.txt"
    output_file_val_4 = "../AI_city_dataset/AICity_after_precessing/validation/market_4_val.txt"

    input_file_val_5 = "../AI_city_dataset/S008/c041/label.txt"
    output_file_val_5 = "../AI_city_dataset/AICity_after_precessing/validation/hallway_1_val.txt"

    input_file_val_6 = "../AI_city_dataset/S008/c042/label.txt"
    output_file_val_6 = "../AI_city_dataset/AICity_after_precessing/validation/hallway_2_val.txt"

    input_file_val_7 = "../AI_city_dataset/S008/c043/label.txt"
    output_file_val_7 = "../AI_city_dataset/AICity_after_precessing/validation/hallway_3_val.txt"

    input_file_val_8 = "../AI_city_dataset/S008/c044/label.txt"
    output_file_val_8 = "../AI_city_dataset/AICity_after_precessing/validation/hallway_4_val.txt"

    input_file_val_9 = "../AI_city_dataset/S013/c071/label.txt"
    output_file_val_9 = "../AI_city_dataset/AICity_after_precessing/validation/hospital_1_val.txt"

    input_file_val_10 = "../AI_city_dataset/S013/c072/label.txt"
    output_file_val_10 = "../AI_city_dataset/AICity_after_precessing/validation/hospital_2_val.txt"

    input_file_val_11 = "../AI_city_dataset/S013/c073/label.txt"
    output_file_val_11 = "../AI_city_dataset/AICity_after_precessing/validation/hospital_3_val.txt"

    input_file_val_12 = "../AI_city_dataset/S013/c074/label.txt"
    output_file_val_12 = "../AI_city_dataset/AICity_after_precessing/validation/hospital_4_val.txt"

    input_file_val_13 = "../AI_city_dataset/S017/c094/label.txt"
    output_file_val_13 = "../AI_city_dataset/AICity_after_precessing/validation/factory_1_val.txt"

    input_file_val_14 = "../AI_city_dataset/S017/c095/label.txt"
    output_file_val_14 = "../AI_city_dataset/AICity_after_precessing/validation/factory_2_val.txt"

    input_file_val_15 = "../AI_city_dataset/S017/c096/label.txt"
    output_file_val_15 = "../AI_city_dataset/AICity_after_precessing/validation/factory_3_val.txt"

    input_file_val_16 = "../AI_city_dataset/S017/c097/label.txt"
    output_file_val_16 = "../AI_city_dataset/AICity_after_precessing/validation/factory_4_val.txt"

    # tesing
    input_file_test_1 = "../AI_city_dataset/S002/c012/label.txt"
    output_file_test_1 = "../AI_city_dataset/AICity_after_precessing/test/market_1_test.txt"

    input_file_test_2 = "../AI_city_dataset/S008/c045/label.txt"
    output_file_test_2 = "../AI_city_dataset/AICity_after_precessing/test/hallway_1_test.txt"

    input_file_test_3 = "../AI_city_dataset/S013/c075/label.txt"
    output_file_test_3 = "../AI_city_dataset/AICity_after_precessing/test/hospital_1_test.txt"

    input_file_test_4 = "../AI_city_dataset/S017/c098/label.txt"
    output_file_test_4 = "../AI_city_dataset/AICity_after_precessing/test/factory_1_test.txt"

    # training
    # result_1 = loading_data(input_file=input_file_1, output_file=output_file_1)
    # result_2 = loading_data(input_file=input_file_2, output_file=output_file_2)
    # result_3 = loading_data(input_file=input_file_3, output_file=output_file_3)
    # result_4 = loading_data(input_file=input_file_4, output_file=output_file_4)
    # result_5 = loading_data(input_file=input_file_5, output_file=output_file_5)
    # result_6 = loading_data(input_file=input_file_6, output_file=output_file_6)
    # result_7 = loading_data(input_file=input_file_7, output_file=output_file_7)
    # result_8 = loading_data(input_file=input_file_8, output_file=output_file_8)
    # result_9 = loading_data(input_file=input_file_9, output_file=output_file_9)
    # result_10 = loading_data(input_file=input_file_10, output_file=output_file_10)
    # result_11 = loading_data(input_file=input_file_11, output_file=output_file_11)
    # result_12 = loading_data(input_file=input_file_12, output_file=output_file_12)
    # result_13 = loading_data(input_file=input_file_13, output_file=output_file_13)
    # result_14 = loading_data(input_file=input_file_14, output_file=output_file_14)
    # result_15 = loading_data(input_file=input_file_15, output_file=output_file_15)
    # result_16 = loading_data(input_file=input_file_16, output_file=output_file_16)

    # result_17 = loading_data(input_file=input_file_17, output_file=output_file_17)
    # result_18 = loading_data(input_file=input_file_18, output_file=output_file_18)
    # result_19 = loading_data(input_file=input_file_19, output_file=output_file_19)
    # result_20 = loading_data(input_file=input_file_20, output_file=output_file_20)
    # result_21 = loading_data(input_file=input_file_21, output_file=output_file_21)
    # result_22 = loading_data(input_file=input_file_22, output_file=output_file_22)
    # result_23 = loading_data(input_file=input_file_23, output_file=output_file_23)
    # result_24 = loading_data(input_file=input_file_24, output_file=output_file_24)
    # result_25 = loading_data(input_file=input_file_25, output_file=output_file_25)
    # result_26 = loading_data(input_file=input_file_26, output_file=output_file_26)
    # result_27 = loading_data(input_file=input_file_27, output_file=output_file_27)
    # result_28 = loading_data(input_file=input_file_28, output_file=output_file_28)
    # result_29 = loading_data(input_file=input_file_29, output_file=output_file_29)
    # result_30 = loading_data(input_file=input_file_30, output_file=output_file_30)
    # result_31 = loading_data(input_file=input_file_31, output_file=output_file_31)
    # result_32 = loading_data(input_file=input_file_32, output_file=output_file_32)

    # validation
    # result_val_1 = loading_data(input_file=input_file_val_1, output_file=output_file_val_1)
    # result_val_2 = loading_data(input_file=input_file_val_2, output_file=output_file_val_2)
    # result_val_3 = loading_data(input_file=input_file_val_3, output_file=output_file_val_3)
    # result_val_4 = loading_data(input_file=input_file_val_4, output_file=output_file_val_4)
    # result_val_5 = loading_data(input_file=input_file_val_5, output_file=output_file_val_5)
    # result_val_6 = loading_data(input_file=input_file_val_6, output_file=output_file_val_6)
    # result_val_7 = loading_data(input_file=input_file_val_7, output_file=output_file_val_7)
    # result_val_8 = loading_data(input_file=input_file_val_8, output_file=output_file_val_8)

    # result_val_9 = loading_data(input_file=input_file_val_9, output_file=output_file_val_9)
    # result_val_10 = loading_data(input_file=input_file_val_10, output_file=output_file_val_10)
    # result_val_11 = loading_data(input_file=input_file_val_11, output_file=output_file_val_11)
    # result_val_12 = loading_data(input_file=input_file_val_12, output_file=output_file_val_12)
    # result_val_13 = loading_data(input_file=input_file_val_13, output_file=output_file_val_13)
    # result_val_14 = loading_data(input_file=input_file_val_14, output_file=output_file_val_14)
    result_val_15 = loading_data(input_file=input_file_val_15, output_file=output_file_val_15)
    # result_val_16 = loading_data(input_file=input_file_val_16, output_file=output_file_val_16)

    # testing
    # result_test_1 = loading_data(input_file=input_file_test_1, output_file=output_file_test_1)
    # result_test_2 = loading_data(input_file=input_file_test_2, output_file=output_file_test_2)

    # result_test_3 = loading_data(input_file=input_file_test_3, output_file=output_file_test_3)
    # result_test_4 = loading_data(input_file=input_file_test_4, output_file=output_file_test_4)

    print("Done!")