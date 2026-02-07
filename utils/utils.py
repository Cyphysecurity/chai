import os
import pandas as pd
def create_csv(round, score, rgb, other_justifications, name, word, gpu, testing, individual, baseline):
    if individual:
        path = "individual_images"
    else:
        path = "training"
    if baseline:
        path = "baseline"
    path = f"stream_frames/{path}/results-{gpu}-testing-{testing}-baseline-{baseline}.csv"
    check_file = os.path.isfile(path)
    colors = f"({rgb[0]}, {rgb[1]}, {rgb[2]})"
    if len(other_justifications) == 1:
        other_justifications.append("No reason given or second vehicle not detected")
    data = {
        'Round': round,
        'Score': score,
        'RGB': [colors],
        'Words': [word],
        'image': name['image'],
        'max_iters': name['max_iters'],
    }
    df = pd.DataFrame(data)
    if check_file:
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

def creat_summary_csv(rgb_letter, word, image_name, result, gpu, testing, testing_ds, max_iters, change=None, llm="None", attack_bool=False, baseline=False):
    if testing:
        path = "testing"
    else:
        path = "training"
    if baseline:
        path = "baseline"
    # Create a DataFrame
    path = f"stream_frames/{path}/summary-training-{gpu}-testing-{testing}-testds-{testing_ds}-changing-{change}-attack-{attack_bool}-{llm}.csv"
    check_file = os.path.isfile(path)
    data = {
        'Image': [image_name],
        'Result': [result],
        'B': [rgb_letter[0]],
        'G': [rgb_letter[1]],
        'R': [rgb_letter[2]],
        'Words': [word],
        'max_iters': [max_iters],
    }
    df = pd.DataFrame(data)
    if check_file:
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

def creat_int_summary_csv(bgr_letter, word, iteration, result, image_name, gpu, testing, testing_ds, max_iters, change=None, llm="None", attack_bool=False, baseline=False):
    # Create a DataFrame
    if testing:
        path = "testing"
    else:
        path = "training"
    if baseline:
        path = "baseline"
    path = f"stream_frames/{path}/summary-int-training-{gpu}-testing-{testing}-testds-{testing_ds}-changing-{change}-attack-{attack_bool}-{llm}.csv"
    check_file = os.path.isfile(path)
    # optimal = get_optimal(image_name)
    data = {
        'Iteration': [iteration],
        'Result': [result],
        'B': [bgr_letter[0]],
        'G': [bgr_letter[1]],
        'R': [bgr_letter[2]],
        'Words': [word],
        'image': image_name,
        'max_iters': [max_iters]
    }
    df = pd.DataFrame(data)
    if check_file:
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

