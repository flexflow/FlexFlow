import os, re, torch
import numpy as np
abs_dirname = os.path.dirname(os.path.abspath(__file__))
hf_path = os.path.join(abs_dirname, "hf_peft_tensors")
ff_path = os.path.join(os.path.dirname(os.path.dirname(abs_dirname)), "build", "inference_tensors")
def print_unique_files_list(dirname):
    files_list = os.listdir(dirname)
    for f in sorted(files_list):
        match = re.search(r'layers.\d+', f)
        if match:
            if "layers." in match[0]:
                layer_num = int(match[0].split(".")[1])
                if layer_num > 0:
                    files_list.remove(f)
            elif "layers_" in match[0]:
                layer_num = int(match[0].split("_")[1])
                if layer_num > 0:
                    files_list.remove(f)
    return sorted(files_list)
def compare_tensors(hf_tensor_filepath, ff_tensor_filepath, tolerance=1e-2):
    if not (os.path.exists(hf_tensor_filepath) and os.path.exists(ff_tensor_filepath)):
        print(hf_tensor_filepath, os.path.exists(hf_tensor_filepath))
        print(ff_tensor_filepath, os.path.exists(ff_tensor_filepath))
        assert False
    hf_tensor = torch.load(hf_tensor_filepath)
    if type(hf_tensor) == tuple or type(hf_tensor) == list:
        assert(len(hf_tensor) == 1)
        hf_tensor = hf_tensor[0]
    hf_tensor = torch.nan_to_num(hf_tensor)
    hf_tensor = hf_tensor.flatten().detach().cpu().numpy()
    ff_tensor = np.loadtxt(ff_tensor_filepath, delimiter=',')

    len_hf_tensor = hf_tensor.shape[0]
    ff_tensor = ff_tensor[:len_hf_tensor]
    
    mismatches = []
    if not np.allclose(ff_tensor, hf_tensor, atol=tolerance):
        print(f"mismatch between {hf_tensor_filepath} and {ff_tensor_filepath}")
        print(f"HF: {hf_tensor}\nFF:{ff_tensor}")
        print(np.isclose(ff_tensor, hf_tensor, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor, hf_tensor, atol=tolerance))[0]
        print(mismatches)
        #print(np.nonzero(hf_tensor)[0])
        # print(np.where(np.isclose(ff_tensor, hf_tensor, atol=tolerance) ==0)[0])
        # print(ff_tensor[36], hf_tensor[36])
    #assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert(len(mismatches) <= .05*len_hf_tensor)
    print("Ok!")
def compare_tensors_difference(hf_tensor_filepath, ff_tensor1_filepath, ff_tensor2_filepath, tolerance=1e-2):
    assert(os.path.exists(hf_tensor_filepath))
    assert(os.path.exists(ff_tensor1_filepath))
    assert(os.path.exists(ff_tensor2_filepath))
    hf_tensor = torch.load(hf_tensor_filepath)
    if type(hf_tensor) == tuple or type(hf_tensor) == list:
        assert(len(hf_tensor) == 1)
        hf_tensor = hf_tensor[0]
    hf_tensor = torch.nan_to_num(hf_tensor)
    hf_tensor = hf_tensor.flatten().detach().cpu().numpy()
    ff_tensor1 = np.loadtxt(ff_tensor1_filepath, delimiter=',')
    ff_tensor2 = np.loadtxt(ff_tensor2_filepath, delimiter=',')

    len_hf_tensor = hf_tensor.shape[0]
    ff_tensor1 = ff_tensor1[:len_hf_tensor]
    ff_tensor2 = ff_tensor2[:len_hf_tensor]
    ff_tensor = ff_tensor1 - ff_tensor2
    
    mismatches = []
    if not np.allclose(ff_tensor, hf_tensor, atol=tolerance):
        print(f"mismatch between {hf_tensor_filepath} and {ff_tensor1_filepath} - {ff_tensor2_filepath}")
        print(f"HF: {hf_tensor}\nFF:{ff_tensor}")
        print(np.isclose(ff_tensor, hf_tensor, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor, hf_tensor, atol=tolerance))[0]
        print(mismatches)
        #print(np.nonzero(hf_tensor)[0])
        # print(np.where(np.isclose(ff_tensor, hf_tensor, atol=tolerance) ==0)[0])
        # print(ff_tensor[36], hf_tensor[36])
    #assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert(len(mismatches) <= .05*len_hf_tensor)
    print("Ok!")
def compare_hf_tensors(tensor1_fp, tensor2_fp):
    assert(os.path.exists(tensor1_fp) and os.path.exists(tensor2_fp))
    hf_tensor1 = torch.load(tensor1_fp)
    hf_tensor2 = torch.load(tensor2_fp)
    if type(hf_tensor1) == tuple or type(hf_tensor1) == list:
        assert(len(hf_tensor1) == 1)
        hf_tensor1 = hf_tensor1[0]
    if type(hf_tensor2) == tuple or type(hf_tensor2) == list:
        assert(len(hf_tensor2) == 1)
        hf_tensor2 = hf_tensor2[0]
    assert(torch.squeeze(hf_tensor1).shape == torch.squeeze(hf_tensor2).shape)
    hf_tensor1 = torch.nan_to_num(hf_tensor1)
    hf_tensor2 = torch.nan_to_num(hf_tensor2)
    if not (np.allclose(hf_tensor1.detach().cpu().numpy(), hf_tensor2.detach().cpu().numpy())):
        print(f"mismatch between {tensor1_fp} and {tensor2_fp}")
        print(hf_tensor1)
        print(hf_tensor2)
        print(np.isclose(hf_tensor1.detach().cpu().numpy(), hf_tensor2.detach().cpu().numpy()))
        mismatches = np.where(~np.isclose(hf_tensor1.detach().cpu().numpy(), hf_tensor2.detach().cpu().numpy()))[0]
        print(mismatches)
        assert(False)
    print("Ok!")

def check_hf_sum_tensors(tensor_sum_fp, tensor1_fp, tensor2_fp):
    assert(os.path.exists(tensor_sum_fp) and os.path.exists(tensor1_fp) and os.path.exists(tensor2_fp))
    hf_tensor_sum = torch.load(tensor_sum_fp)
    hf_tensor1 = torch.load(tensor1_fp)
    hf_tensor2 = torch.load(tensor2_fp)
    if type(hf_tensor_sum) == tuple or type(hf_tensor_sum) == list:
        assert(len(hf_tensor_sum) == 1)
        hf_tensor_sum = hf_tensor_sum[0]
    if type(hf_tensor1) == tuple or type(hf_tensor1) == list:
        assert(len(hf_tensor1) == 1)
        hf_tensor1 = hf_tensor1[0]
    if type(hf_tensor2) == tuple or type(hf_tensor2) == list:
        assert(len(hf_tensor2) == 1)
        hf_tensor2 = hf_tensor2[0]
    assert(torch.squeeze(hf_tensor_sum).shape == torch.squeeze(hf_tensor1).shape)
    assert(torch.squeeze(hf_tensor1).shape == torch.squeeze(hf_tensor2).shape)
    hf_tensor1 = torch.nan_to_num(hf_tensor1)
    hf_tensor2 = torch.nan_to_num(hf_tensor2)
    hf_tensor_sum = torch.nan_to_num(hf_tensor_sum)
    sum_check_tensor = hf_tensor1 + hf_tensor2
    if not (np.allclose(sum_check_tensor.detach().cpu().numpy(), hf_tensor_sum.detach().cpu().numpy())):
        print(f"mismatch between {sum_check_tensor} and {tensor1_fp} + {tensor2_fp}")
        print(tensor_sum_fp)
        print(sum_check_tensor)
        print(hf_tensor1)
        print(hf_tensor2)
        print(np.isclose(sum_check_tensor.detach().cpu().numpy(), hf_tensor_sum.detach().cpu().numpy()))
        mismatches = np.where(~np.isclose(sum_check_tensor.detach().cpu().numpy(), hf_tensor_sum.detach().cpu().numpy()))[0]
        print(mismatches)
        assert(False)
    print("Ok!")
def check_hf_zero_tensor(hf_tensor_fp):
    assert(os.path.exists(hf_tensor_fp))
    hf_tensor1 = torch.load(hf_tensor_fp)
    if type(hf_tensor1) == tuple or type(hf_tensor1) == list:
        assert(len(hf_tensor1) == 1)
        hf_tensor1 = hf_tensor1[0]
    assert(torch.count_nonzero(torch.nan_to_num(hf_tensor1)).sum() == 0)
def print_tensors(hf_tensor_filepath, ff_tensor_filepath, txt=""):
    assert(os.path.exists(hf_tensor_filepath) and os.path.exists(ff_tensor_filepath))
    hf_tensor = torch.load(hf_tensor_filepath)
    if type(hf_tensor) == tuple or type(hf_tensor) == list:
        assert(len(hf_tensor) == 1)
        hf_tensor = hf_tensor[0]
    hf_tensor = torch.nan_to_num(hf_tensor)
    hf_tensor = hf_tensor.flatten().detach().cpu().numpy()
    ff_tensor = np.loadtxt(ff_tensor_filepath, delimiter=',')

    len_hf_tensor = hf_tensor.shape[0]
    ff_tensor = ff_tensor[:len_hf_tensor]

    print(f"{txt} - HF tensor:")
    print(hf_tensor)
    print(f"{txt} - FF tensor: ")
    print(ff_tensor)
def compare_flexflow_tensors(ff_tensor1_fp, ff_tensor2_fp, tolerance=1e-5, max_len=-1):
    assert(os.path.exists(ff_tensor1_fp) and os.path.exists(ff_tensor2_fp))
    ff_tensor1 = np.loadtxt(ff_tensor1_fp, delimiter=',')
    ff_tensor2 = np.loadtxt(ff_tensor2_fp, delimiter=',')

    if (ff_tensor1.shape != ff_tensor2.shape):
        print(ff_tensor1.shape, ff_tensor2.shape)
    assert(ff_tensor1.shape == ff_tensor2.shape)

    if max_len > -1:
        ff_tensor1 = ff_tensor1[:max_len]
        ff_tensor2 = ff_tensor2[:max_len]
    
    mismatches = []
    if not np.allclose(ff_tensor1, ff_tensor2, atol=tolerance):
        print(f"mismatch between {ff_tensor1_fp} and {ff_tensor2_fp}")
        print(f"Tensor1: {ff_tensor1}\nTensor2:{ff_tensor2}")
        print(np.isclose(ff_tensor1, ff_tensor2, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor1, ff_tensor2, atol=tolerance))[0]
        print(mismatches)
    #assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert(len(mismatches) <= .05*len(ff_tensor1))
    print("Ok!")
def compare_flexflow_tensors_shortest(ff_tensor1_fp, ff_tensor2_fp, tolerance=1e-5):
    assert(os.path.exists(ff_tensor1_fp) and os.path.exists(ff_tensor2_fp))
    ff_tensor1 = np.loadtxt(ff_tensor1_fp, delimiter=',')
    ff_tensor2 = np.loadtxt(ff_tensor2_fp, delimiter=',')
    minlen = min(ff_tensor1.shape[0], ff_tensor2.shape[0])
    ff_tensor1 = ff_tensor1[:minlen]
    ff_tensor2 = ff_tensor2[:minlen]
    mismatches = []
    if not np.allclose(ff_tensor1, ff_tensor2, atol=tolerance):
        print(f"mismatch between {ff_tensor1_fp} and {ff_tensor2_fp}")
        print(f"Tensor1: {ff_tensor1}\nTensor2:{ff_tensor2}")
        print(np.isclose(ff_tensor1, ff_tensor2, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor1, ff_tensor2, atol=tolerance))[0]
        print(mismatches)
    #assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert(len(mismatches) <= .05*len(ff_tensor1))
    print("Ok!")
def check_flexflow_tensors_sum(ff_tensor_sum_fp, ff_tensor1_fp, ff_tensor2_fp, tolerance=1e-5):
    assert(os.path.exists(ff_tensor1_fp) and os.path.exists(ff_tensor2_fp))
    ff_tensor1 = np.loadtxt(ff_tensor1_fp, delimiter=',')
    ff_tensor2 = np.loadtxt(ff_tensor2_fp, delimiter=',')
    ff_tensor_sum = np.loadtxt(ff_tensor_sum_fp, delimiter=',')
    
    ff_sum = ff_tensor1 + ff_tensor2
    assert(ff_tensor1.shape == ff_tensor2.shape)
    
    mismatches = []
    if not np.allclose(ff_tensor_sum, ff_sum, atol=tolerance):
        print(f"mismatch between {ff_tensor_sum_fp} and sum of {ff_tensor1_fp} + {ff_tensor2_fp}")
        print(f"Tensor1: {ff_tensor1}\nTensor2:{ff_tensor2}")
        print(f"Sum Tensor: {ff_tensor_sum}\nActual sum:{ff_sum}")
        print(np.isclose(ff_tensor_sum, ff_sum, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor_sum, ff_sum, atol=tolerance))[0]
        print(mismatches)
    #assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert(len(mismatches) <= .05*len(ff_tensor1))
    print("Ok!")
def load_ff_tensor(filename, shape):
    if ff_path not in filename:
        filename = os.path.join(ff_path, filename)
    ff_tensor = np.loadtxt(filename, delimiter=',').reshape(shape, order = 'F')
    return ff_tensor
def load_hf_tensor(filename):
    if hf_path not in filename:
        filename = os.path.join(hf_path, filename)
    hf_tensor = torch.load(filename)
    hf_tensor = hf_tensor.detach().cpu().numpy()
    return hf_tensor