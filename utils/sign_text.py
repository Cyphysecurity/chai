"""
Given an image with a sign, designed by four corners, put some text on it. This script
generates the text with word wrapping, then performs a perspective warp to overlay the text
onto the sign. Use the warp_text_onto_sign function to overlay the text onto the sign,
which does everything and is the only function that should be called from outside this file.
"""

import cv2
import numpy as np
from PIL import Image


def _wrap_text(text, font, font_scale, thickness, max_width):
    """
    Splits text into lines so that each line's width does not exceed max_width.
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        candidate_line = word if current_line == "" else current_line + " " + word
        (line_width, _), _ = cv2.getTextSize(candidate_line, font, font_scale, thickness)
        if line_width > max_width and current_line != "":
            lines.append(current_line)
            current_line = word
        else:
            current_line = candidate_line
    if current_line:
        lines.append(current_line)
    return lines


def _compute_text_block_size(lines, font, font_scale, thickness, line_spacing=1.2):
    """
    Computes the maximum line width and total height for a list of text lines.
    """
    widths = []
    heights = []
    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        widths.append(w)
        heights.append(h + baseline)
    total_height = int(sum(heights) + (len(lines) - 1) * (line_spacing - 1) * (max(heights) if heights else 0))
    max_width = max(widths) if widths else 0
    return max_width, total_height, heights


def _find_max_font_scale(text1, text2, font, thickness, allowed_width, allowed_height,
                         line_spacing=1.2, scale_min=0.1, scale_max=10.0, tol=0.01):
    """
    Uses binary search to find the maximum font scale such that the wrapped text
    fits within allowed_width and allowed_height.
    """
    best_scale = scale_min
    while scale_max - scale_min > tol:
        mid_scale = (scale_max + scale_min) / 2.0
        lines = [text1, text2] if text2 else _wrap_text(text1, font, mid_scale, thickness, allowed_width)
        text_w, text_h, _ = _compute_text_block_size(lines, font, mid_scale, thickness, line_spacing)
        if text_w <= allowed_width and text_h <= allowed_height:
            best_scale = mid_scale
            scale_min = mid_scale  # try a larger scale
        else:
            scale_max = mid_scale  # too big, make it smaller
    final_lines = [text1, text2] if text2 else _wrap_text(text1, font, best_scale, thickness, allowed_width)
    return best_scale, final_lines


def _create_text_image(text1, text2, font, thickness, canvas_width, canvas_height, pad_x, pad_y,
                       line_spacing=1.2, text_color=(0, 0, 255, 255), text_bck = (255,255,255,255)):
    """
    Creates a BGRA image with transparent background containing the wrapped text.
    The text is centered within a padded region of the canvas.
    """
    allowed_width = canvas_width - 2 * pad_x
    allowed_height = canvas_height - 2 * pad_y

    best_scale, lines = _find_max_font_scale(text1, text2, font, thickness, allowed_width, allowed_height, line_spacing)
    text_w, text_h, line_heights = _compute_text_block_size(lines, font, best_scale, thickness, line_spacing)

    # Create a BGRA image with transparent background.
    # Note that we set the background's color to the text color with alpha 0 (fully transparent).
    # Not doing this (i.e., setting the background to black) causes some black fringe around the text, perhaps
    # related to antialiasing during the warp operation where these "black" transparent pixels may be referenced.
    text_img = np.full((canvas_height, canvas_width, 4), (text_color[0], text_color[1], text_color[2], 0),
                       dtype=np.uint8)

    # Starting y: center vertically within the allowed area.
    y_offset = pad_y + (allowed_height - text_h) // 2 + line_heights[0]

    for i, line in enumerate(lines):
        (line_w, line_h), baseline = cv2.getTextSize(line, font, best_scale, thickness)
        # Center each line horizontally.
        x_offset = pad_x + (allowed_width - line_w) // 2

        bottom_left = (x_offset, y_offset)
        bottom_left = (bottom_left[0], bottom_left[1] + line_h//2 + 10)
        top_right = (x_offset + line_w, y_offset - line_h - 10)

        # Draw the background rectangle
        if text_bck is not None:
            cv2.rectangle(text_img, bottom_left, top_right, text_bck, thickness=cv2.FILLED)
        cv2.putText(text_img, line, (x_offset, y_offset), font, best_scale, text_color, thickness, cv2.LINE_AA)
        # Increment y_offset by the height of the line and extra spacing.
        y_offset += int(line_heights[i] * line_spacing)

    return text_img


def warp_text_onto_sign(image, text1, text2, sign_corners,
                        text_color=(0, 0, 255),
                        font=cv2.FONT_HERSHEY_SIMPLEX,
                        thickness=10,
                        high_res_factor=3,
                        line_spacing=1.2,
                        pad_ratio=0.05,
                        bck_color=None):
    """
    Overlays wrapped text onto the sign.

    Parameters:
      - image: Destination BGR image.
      - text: The text to overlay.
      - sign_corners: List of four (x, y) tuples (top-left, top-right, bottom-right, bottom-left).
      - text_color: BGR color tuple. Importantly, this is BGR not RGB. Alpha is set to 255 (fully opaque).
      - font: OpenCV font.
      - thickness: Text thickness.
      - high_res_factor: Factor by which the text image is higher resolution than the sign box (higher = sharper text).
      - line_spacing: Multiplier for spacing between lines.
      - pad_ratio: Fraction of canvas dimensions used for padding.

    Returns:
      The image with the warped text composited onto it.
    """
    # sign_corners = [(x+150, y+150) for (x,y) in sign_corners]
    sign_corners_arr = np.array(sign_corners, dtype=np.float32)

    # We can approximate the sign's dimensions by computing the average width and height
    # for the projected sign's corners. This isn't really a good way, since if the sign is
    # very skewed, the height to width ratio of the skewed sign may be very different from the
    # true height to width ratio of the sign. This could be solved by having the true 3D coordinates
    # of the sign, or at least having the true height and width of the sign. But this works well
    # enough for signs that aren't too skewed.
    width_top = np.linalg.norm(sign_corners_arr[1] - sign_corners_arr[0])
    width_bottom = np.linalg.norm(sign_corners_arr[2] - sign_corners_arr[3])
    sign_width = (width_top + width_bottom) / 2.0

    height_left = np.linalg.norm(sign_corners_arr[3] - sign_corners_arr[0])
    height_right = np.linalg.norm(sign_corners_arr[2] - sign_corners_arr[1])
    sign_height = (height_left + height_right) / 2.0

    canvas_width = int(sign_width * 5)
    canvas_height = int(sign_height * 5)
    pad_x = int(canvas_width * pad_ratio)
    pad_y = int(canvas_height * pad_ratio)

    text_color_bgra = (text_color[0], text_color[1], text_color[2], 255)  # Full opacity

    text_img = _create_text_image(text1, text2, font, thickness, canvas_width, canvas_height, pad_x, pad_y,
                                  line_spacing, text_color_bgra, bck_color)

    src_pts = np.array([
        [0, 0],
        [canvas_width - 1, 0],
        [canvas_width - 1, canvas_height - 1],
        [0, canvas_height - 1]
    ], dtype=np.float32)
    dst_pts = sign_corners_arr

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_text = cv2.warpPerspective(text_img, M, (image.shape[1], image.shape[0]),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0, 0))

    b, g, r, a = cv2.split(warped_text)
    alpha_mask = a.astype(float) / 255.0
    result = image.astype(float)
    for c, channel in enumerate([b, g, r]):
        result[:, :, c] = channel * alpha_mask + result[:, :, c] * (1 - alpha_mask)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


if __name__ == "__main__":
    # Image with a white background (255 for all 3 color channels).
    image = np.full((500, 800, 3), 255, dtype=np.uint8)

    # Sign's corners in pixel coordinates.
    sign_corners = [(150, 100), (650, 80), (700, 400), (100, 420)]

    # Draw a green rectangle to visualize the sign's corners.
    for i in range(4):
        pt1 = tuple(sign_corners[i])
        pt2 = tuple(sign_corners[(i + 1) % 4])
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)  # Draw green lines for borders

    # The text to overlay.
    # text = "Hello, World!"
    text = "The quick brown fox jumped over the lazy dog."
    # text = ("The cat (Felis catus), also referred to as the domestic cat, is a small domesticated carnivorous "
    #         "mammal. It is the only domesticated species of the family Felidae. Advances in archaeology and "
    #         "genetics have shown that the domestication of the cat occurred in the Near East around 7500 BC.")

    output = warp_text_onto_sign(image, text, sign_corners, text_color=(240, 240, 240), thickness=10)

    cv2.imwrite("output.png", output)

    cv2.imshow("Warped Sign Text", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
