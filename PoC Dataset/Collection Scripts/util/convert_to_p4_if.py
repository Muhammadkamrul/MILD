INPUT_FILE = "modified_rules.txt"
OUTPUT_FILE = "generated_code/ml_surrogate_model_rules.p4.part"

def convert_to_p4_syntax(input_file, output_file):
    """
    Converts Python if statements to C syntax with specified variable name changes.
    This version handles the specific format of the input file, where each if and else creates
    a new level of nesting regardless of indentation.
    Also multiplies numeric values by 100 and truncates the decimal part.
    
    Args:
        input_file (str): Path to input file with Python if statements
        output_file (str): Path to output file for C if statements
    """
    # Define mappings for variable name changes
    variable_name_mappings = {
        "Prefix Inter-packet Time": "prefix_inter_packet_time",
        "Inter-packet Time": "inter_packet_time",
        "hopCount": "hop_count",
        "New Addresses Seen": "new_addresses_seen",
        "Port": "port",
        "Source": "src_ipv6_addr",
        "Protocol": "protocol",
    }
    
    # Define detailed action mappings with multi-line if statements
    action_mappings = {
        "Drop Internal Flooding": """// Internal Flooding
if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal == 1000) {
    internal_flooding_tcp_benign_drop();
} else if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal != 1000) {
    internal_flooding_tcp_attack_drop();
} else {
    internal_flooding_ns_drop();
}""",

        "Drop Internal Spoofing": """// Internal Spoofing
if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal == 1000) {
    internal_spoofing_tcp_benign_drop();
} else if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal != 1000) {
    internal_spoofing_tcp_attack_drop();
} else {
    internal_spoofing_ns_drop();
}""",

        "Drop DDOS": """// External Flooding Spoofing
if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal == 1000) {
    external_flooding_spoofing_tcp_benign_drop();
} else if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal != 1000) {
    external_flooding_spoofing_tcp_attack_drop();
} else {
    external_flooding_spoofing_ns_drop();
}""",

        "Drop External Spoofing": """// External Spoofing
if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal == 1000) {
    external_spoofing_tcp_benign_drop();
} else if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal != 1000) {
    external_spoofing_tcp_attack_drop();
} else {
    external_spoofing_ns_drop();
}""",

        "Drop External Flooding": """// External Flooding
if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal == 1000) {
    external_flooding_tcp_benign_drop();
} else if (hdr.tcp.isValid() && hdr.tcp_option_ts.isValid() && hdr.tcp_option_ts.tsVal != 1000) {
    external_flooding_tcp_attack_drop();
} else if (hdr.icmpv6.isValid() && hdr.icmpv6.type == TYPE_ICMPV6_NS) {
    external_flooding_ns_drop();
}"""
    }

    # Read the input file
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Stack to keep track of the structure
    stack = []
    c_code = []
    
    for line in lines:
        if line.startswith("if "):
            # Extract and convert condition
            condition = line[3:].rstrip(':')
            
            # Apply name mappings
            for py_name, c_name in variable_name_mappings.items():
                condition = condition.replace(py_name, c_name)
            
            # Multiply numeric values by 100 and truncate
            condition = multiply_and_truncate_numbers(condition)
            
            # Add C-style if statement with proper indentation
            c_code.append("    " * len(stack) + f"if ({condition}) {{")
            stack.append("if")
        
        elif line.startswith("else:"):
            # Pop the previous statement from the stack
            popped = None
            if stack:
                popped = stack.pop()
            
            # Add C-style else statement
            if popped and popped == "if":    
                c_code.append("    " * len(stack) + "} else {")
            elif popped and popped == "else":
                c_code.append("    " * len(stack) + "}")
                while True:
                    if len(stack) == 0:
                        break

                    popped = stack.pop()
                    if popped == "if":
                        line = "    " * len(stack) + "} else {"
                    elif popped == "else":
                        line = "    " * len(stack) + "}"
                    
                    c_code.append(line)
                    if line.endswith("} else {"):
                        break
                
            stack.append("else")
        
        elif line.startswith("#"):
            # Comment line
            comment = line[1:].strip()
            c_code.append("    " * len(stack) + f"/* {comment} */")
        
        else:
            # Action line - check if it matches any of our action mappings
            if line in action_mappings:
                # Split the multi-line action into lines
                action_lines = action_mappings[line].split('\n')
                # Add each line with proper indentation
                for action_line in action_lines:
                    c_code.append("    " * len(stack) + action_line)
            else:
                # Default case - just add as a comment
                c_code.append("    " * len(stack) + f"// {line}")
    
    # Close any remaining blocks
    while stack:
        stack.pop()
        c_code.append("    " * len(stack) + "}")
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write("\n".join(c_code))
    
    print(f"Conversion complete. C code written to {output_file}")

def multiply_and_truncate_numbers(condition):
    """
    Finds numerical values in the condition, multiplies them by 1000, and truncates the decimal part.
    
    Args:
        condition (str): The condition string from an if statement
    
    Returns:
        str: The condition with numerical values multiplied by 1000 and truncated
    """
    import re
    
    # This pattern matches comparison operators (<=, >=, ==, !=, <, >) followed by a space and a number
    pattern = r'(<=|>=|==|!=|<|>)\s*(-?\d+\.?\d*)'
    
    def replace_match(match):
        operator = match.group(1)
        number = float(match.group(2))
        # Multiply by 1000 and truncate (remove decimal part)
        new_number = int(number * 1000 * 1000)
        return f"{operator} {new_number}"
    
    # Replace numbers in the condition
    modified_condition = re.sub(pattern, replace_match, condition)
    return modified_condition

if __name__ == "__main__":
    convert_to_p4_syntax(INPUT_FILE, OUTPUT_FILE)