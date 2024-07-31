from PIL import Image, ImageDraw, ImageFont

import numpy as np
import cv2


colors = [
    [127, 255, 0],   # Bright Lime
    [255, 128, 0],   # Bright Orange
    [0, 127, 255],   # Bright Sky Blue
    [255, 0, 127],   # Bright Pink
    [127, 0, 255],   # Bright Violet
    [255, 127, 0],   # Bright Amber
    [0, 255, 127],   # Bright Spring Green
    [255, 0, 0],     # Bright Red
    [0, 255, 0],     # Bright Green
    [0, 0, 255],     # Bright Blue
    [255, 255, 0],   # Bright Yellow
    [0, 255, 255],   # Bright Cyan
    [255, 0, 255],   # Bright Magenta
]

def make_color_lighter(color, percentage=0.5):
    # Ensure the percentage is between 0 and 1
    percentage = max(min(percentage, 1), 0)
    
    # Parse the color string into an RGB tuple
    rgb = tuple(map(int, color.strip('[]').split(',')))
    
    # Calculate the new color by moving the original color towards white by the given percentage
    lighter_rgb = tuple(int((1 - percentage) * c + percentage * 255) for c in rgb)
    return lighter_rgb

def draw_text_on_image_with_score(sentence, entities, ei_scores_text, top=1):
    # Store high light dict
    topk_scores, topk_tokens = ei_scores_text.topk(top, dim=-1)
    highlight_dict = {}
    for idx, token_ids in enumerate(topk_tokens):
        color = colors[idx]
        text_list = [entities[token_id.item()].text for token_id in token_ids]
        index_list = [[sentence.index(_str), sentence.index(_str)+len(_str)] for _str in text_list]
        index_list.sort(key=lambda x: x[0])
        highlight_dict[str(color)] = index_list

    sorted_highlight_items = sorted(highlight_dict.items(), key=lambda item: item[1][0][0])
    highlight_dict = {k: v for k, v in sorted_highlight_items}

    # Load a high-quality truetype font
    font_path = "./demo/find/arial.ttf"  # Update this to the path of the font file you want to use
    font_size = 20           # You can adjust this size to your preference

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Font file not found. Falling back to default font.")
        font = ImageFont.load_default()

    width, height = 350, 250
    image = Image.new('RGB', (width, height), color='white')
    # Initialize the drawing context with the image as background
    draw = ImageDraw.Draw(image)

    # Define starting position
    x, y = 10, 10
    width_limit = width - 20
    space_width = draw.textbbox((0, 0), ' ', font=font)[2]
    words = sentence.split()
    word_lengths = [draw.textbbox((0, 0), word, font=font)[2] for word in words]

    current_line_length = 0
    lines = []
    line = []
    for word, word_length in zip(words, word_lengths):
        if current_line_length + word_length <= width_limit:
            line.append(word)
            current_line_length += word_length + space_width
        else:
            lines.append(' '.join(line))
            line = [word]
            current_line_length = word_length + space_width
    lines.append(' '.join(line))  # Add the last line
    
    for line in lines:
        
        line_length = draw.textbbox((0,0), line, font=font)[2]
        line_highlight_ranges = []
        
        # Calculate the positions for highlights
        start_idx = sentence.find(line)
        end_idx = start_idx + len(line)
        
        for color, ranges in highlight_dict.items():
            for range_start, range_end in ranges:
                if range_start < end_idx and range_end > start_idx:
                    line_highlight_ranges.append((
                        max(range_start, start_idx) - start_idx,
                        min(range_end, end_idx) - start_idx,
                        color
                    ))
        
        # Draw the highlights
        offset = 0
        for start, end, color in line_highlight_ranges:
            highlight_text = sentence[start_idx + offset:start_idx + start]
            draw.text((x, y), highlight_text, font=font, fill='black')
            x += draw.textbbox((0,0), highlight_text, font=font)[2]
            
            highlight_text = sentence[start_idx + start:start_idx + end]
            color_tuple = make_color_lighter(color)
            draw.rectangle(((x, y), (x + draw.textbbox((0,0), highlight_text, font=font)[2], y + font_size)), fill=color_tuple)
            draw.text((x, y), highlight_text, font=font, fill='black')
            x += draw.textbbox((0,0), highlight_text, font=font)[2]
            offset = end
        
        # Draw the rest of the line
        remaining_text = sentence[start_idx + offset:end_idx]
        draw.text((x, y), remaining_text, font=font, fill='black')
        y += font.getbbox(line)[3]  # Move to the next line
        x = 10  # Reset x position

    # image.save("test.png")
    # import pdb; pdb.set_trace()
    return image

def put_text(image_draw, phrases):
    # image_draw is your actual image.
    # phrases is a list of dictionaries with 'text' and 'color' keys

    # Choose your font
    font = cv2.FONT_HERSHEY_DUPLEX
    # Choose the size of your font
    font_scale = 0.4
    # Choose the thickness of the font
    thickness = 1

    # set the text start position
    text_offset_x = 10
    text_offset_y = image_draw.shape[0] + 10  # start from 10 pixels below the bottom of the image

    # Initialize some variables for the maximum text width and the total text height
    total_text_height = 10  # start with padding for the bottom of the image

    # Calculate total_text_height
    for phrase in phrases:
        _str = phrase['text']
        (_, text_height) = cv2.getTextSize(_str, font, font_scale, thickness)[0]
        total_text_height += text_height + 10  # add padding between phrases

    # Make a canvas to fit the image and the text
    canvas = np.ones((image_draw.shape[0] + total_text_height, image_draw.shape[1], 3), dtype='uint8') * 248

    # Copy the image to the canvas
    canvas[:image_draw.shape[0], :image_draw.shape[1]] = image_draw

    # Add each phrase to the canvas
    for phrase in phrases:
        _str = phrase['text']
        color = phrase['color']

        # Use getTextSize to get the width and height of the text box
        (_, text_height) = cv2.getTextSize(_str, font, font_scale, thickness)[0]

        # Add text to the canvas
        cv2.putText(canvas, _str, (text_offset_x, text_offset_y + text_height), font, font_scale, color, thickness)

        # Update the y position for next text
        text_offset_y += text_height + 10  # 10 is the vertical space between phrases

    return canvas

def draw_instances(image, instances):
    """
    Draw bounding boxes, overlay masks and add class labels on the image.

    Parameters:
    - image: numpy array of shape (H, W, C)
    - instances: an object with the following attributes
        - gt_masks: a tensor of shape (N, H, W), where N is the number of instances
        - gt_boxes: a tensor of shape (N, 4), where each row is (x1, y1, x2, y2)
        - gt_classes: a list of N class labels
    """

    # For each instance
    for i in range(len(instances.gt_classes)):
        # Get a random color
        color = colors[i]

        # Draw the bounding box
        x1, y1, x2, y2 = instances.gt_boxes.tensor[i].int().numpy()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        color = np.array(color)[None, None, :]
        mask = instances.gt_masks.tensor[i].numpy()[:,:,None]
        overlay = mask * color
        image = image * (1 - mask) + (image * mask * 0.5 + overlay * 0.5)
    return image
