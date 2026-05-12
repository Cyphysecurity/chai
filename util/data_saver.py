
class DataSaver():
    def __init__(self, file_name, targets, name_answer):
        self.header = "n_round;image;score;success;"
        for target in targets:
            self.header += name_answer[target] + ";"
            self.header += name_answer[target] + "_total;"
        self.header += "r_letter;g_letter;b_letter;r_bck;g_bck;b_bck;size;"
        for target in targets:
            self.header += f"optimal_txt_{name_answer[target]};"
        
        self.header += "total_images\n"
        self.text_print = ""
        self.file_name = file_name
    
    def add_data(self, attack_params, score, n_round, image, success, optimal_vps, n_test_images, targets, score_per_target):
        rgb_letter = attack_params["rgb_letter"]
        rgb_background = attack_params["rgb_background"]
        self.text_print+= f"{n_round};{image};{score};{success};"
        for target in targets:
            self.text_print += f"{score_per_target[target][0]};"
            self.text_print += f"{score_per_target[target][1]};"
        self.text_print += f"{rgb_letter[0]};{rgb_letter[1]};{rgb_letter[2]};{rgb_background[0]};{rgb_background[1]};{rgb_background[2]};{round(attack_params['size'],2)}"
        for op_vp in optimal_vps: 
            self.text_print += f";{op_vp}"
        self.text_print += f";{n_test_images}\n"
    
    def save(self):
        with open(self.file_name, "w") as f:
            f.write(self.header + self.text_print)
