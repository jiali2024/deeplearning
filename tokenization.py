import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

input_str ="用简单语言讲解Transformer神经网络架构"
print(f"input_str: {input_str}")
# input_str: 用简单语言讲解Transformer神经网络架构
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens = encoding.encode(input_str)
print(f"tokens: {tokens}")
# tokens: [11883, 99337, 24946, 73981, 78244, 10414, 110, 50338, 47458, 55228, 252, 54493, 72456, 20119, 114, 78935]
num_tokens = num_tokens_from_string("用简单语言讲解Transformer神经网络架构", "cl100k_base")
print(f"num_tokens: {num_tokens}")
# num_tokens: 16
decode_input_str = encoding.decode(tokens)
print(f"decode_input_str: {decode_input_str}")
# decode_input_str: 用简单语言讲解Transformer神经网络架构
bytes_list = [encoding.decode_single_token_bytes(token) for token in tokens]
print(f"bytes_list: {bytes_list}")
# bytes_list: [b'\xe7\x94\xa8', 
# b'\xe7\xae\x80', 
# b'\xe5\x8d\x95', 
# b'\xe8\xaf\xad', 
# b'\xe8\xa8\x80', 
# b'\xe8\xae', b'\xb2', 讲
# b'\xe8\xa7\xa3', 
# b'Transformer', 
# b'\xe7\xa5', b'\x9e', 神
# b'\xe7\xbb\x8f', 
# b'\xe7\xbd\x91\xe7\xbb\x9c', 
# b'\xe6\x9e', b'\xb6', 架
# b'\xe6\x9e\x84']
pre_bytes = b''
for i, bytes in enumerate(bytes_list):
    # Attempt to decode the current bytes
    string = (pre_bytes + bytes).decode("utf-8", "ignore")
    print(f"bytes_list{i}: {(pre_bytes + bytes)} -> {string}")
    if string:
        # If successful, print the result and reset pre_bytes
        pre_bytes = b''
    else:
        # If not successful, append the current bytes to pre_bytes for the next iteration
        pre_bytes += bytes
# bytes_list0: b'\xe7\x94\xa8' -> 用
# bytes_list1: b'\xe7\xae\x80' -> 简
# bytes_list2: b'\xe5\x8d\x95' -> 单
# bytes_list3: b'\xe8\xaf\xad' -> 语
# bytes_list4: b'\xe8\xa8\x80' -> 言
# bytes_list5: b'\xe8\xae' -> 
# bytes_list6: b'\xe8\xae\xb2' -> 讲
# bytes_list7: b'\xe8\xa7\xa3' -> 解
# bytes_list8: b'Transformer' -> Transformer
# bytes_list9: b'\xe7\xa5' -> 
# bytes_list10: b'\xe7\xa5\x9e' -> 神
# bytes_list11: b'\xe7\xbb\x8f' -> 经
# bytes_list12: b'\xe7\xbd\x91\xe7\xbb\x9c' -> 网络
# bytes_list13: b'\xe6\x9e' -> 
# bytes_list14: b'\xe6\x9e\xb6' -> 架
# bytes_list15: b'\xe6\x9e\x84' -> 构