from PIL import Image
import math
import matplotlib.pyplot as plt


def prewitt_edge_detection(input_path):
    img = Image.open(input_path).convert('L')
    width, height = img.size
    pixels = list(img.getdata())
    pixels_2d = [pixels[i * width:(i + 1) * width] for i in range(height)]

    Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    Gy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

    res_x = [[
        sum(Gx[dy][dx] * pixels_2d[y + dy - 1][x + dx - 1] for dy in range(3) for dx in range(3))
        if 1 <= x < width - 1 and 1 <= y < height - 1 else 0
        for x in range(width)] for y in range(height)]

    res_y = [[
        sum(Gy[dy][dx] * pixels_2d[y + dy - 1][x + dx - 1] for dy in range(3) for dx in range(3))
        if 1 <= x < width - 1 and 1 <= y < height - 1 else 0
        for x in range(width)] for y in range(height)]

    edge_magnitude = [[
        min(255, int(math.sqrt(res_x[y][x]**2 + res_y[y][x]**2))) for x in range(width)
    ] for y in range(height)]
    return edge_magnitude


def canny_edge_detection(input_path):
    img = Image.open(input_path).convert('L')
    width, height = img.size
    pixels = list(img.getdata())
    gray_2d = [pixels[i * width:(i + 1) * width] for i in range(height)]

    gaussian_kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    blurred_2d = [[
        sum(gaussian_kernel[dy][dx] * gray_2d[y + dy - 1][x + dx - 1] for dy in range(3) for dx in range(3)) / 16.0
        if 1 <= x < width - 1 and 1 <= y < height - 1 else gray_2d[y][x]
        for x in range(width)] for y in range(height)]

    sobel_gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_gy = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    gx_2d = [[
        sum(sobel_gx[dy][dx] * blurred_2d[y + dy - 1][x + dx - 1] for dy in range(3) for dx in range(3))
        if 1 <= x < width - 1 and 1 <= y < height - 1 else 0
        for x in range(width)] for y in range(height)]

    gy_2d = [[
        sum(sobel_gy[dy][dx] * blurred_2d[y + dy - 1][x + dx - 1] for dy in range(3) for dx in range(3))
        if 1 <= x < width - 1 and 1 <= y < height - 1 else 0
        for x in range(width)] for y in range(height)]

    mag_2d = [[math.sqrt(gx_2d[y][x]**2 + gy_2d[y][x]**2) for x in range(width)] for y in range(height)]
    dir_2d = [[
        math.degrees(math.atan2(gy_2d[y][x], gx_2d[y][x])) % 180 if mag_2d[y][x] != 0 else 0 for x in range(width)
    ] for y in range(height)]

    nms_2d = [[
        mag_2d[y][x] if (
            (0 <= dir_2d[y][x] < 22.5 or 157.5 <= dir_2d[y][x] < 180) and mag_2d[y][x] >= max(mag_2d[y][x - 1], mag_2d[y][x + 1])
            or (22.5 <= dir_2d[y][x] < 67.5) and mag_2d[y][x] >= max(mag_2d[y - 1][x + 1], mag_2d[y + 1][x - 1])
            or (67.5 <= dir_2d[y][x] < 112.5) and mag_2d[y][x] >= max(mag_2d[y - 1][x], mag_2d[y + 1][x])
            or (112.5 <= dir_2d[y][x] < 157.5) and mag_2d[y][x] >= max(mag_2d[y - 1][x - 1], mag_2d[y + 1][x + 1])
        ) else 0
        for x in range(1, width - 1)] for y in range(1, height - 1)]

    max_magnitude = max(max(row) for row in nms_2d)
    high_threshold, low_threshold = 0.25 * max_magnitude, 0.12 * max_magnitude

    dt_2d = [[
        255 if nms_2d[y][x] >= high_threshold else 50 if nms_2d[y][x] >= low_threshold else 0
        for x in range(len(nms_2d[0]))
    ] for y in range(len(nms_2d))]

    final_2d = [[
        255 if dt_2d[y][x] == 255 or (
            dt_2d[y][x] == 50 and any(
                dt_2d[yy][xx] == 255 for yy in range(max(0, y - 1), min(len(dt_2d), y + 2))
                for xx in range(max(0, x - 1), min(len(dt_2d[0]), x + 2))
            )
        ) else 0
        for x in range(len(dt_2d[0]))
    ] for y in range(len(dt_2d))]
    return final_2d


def display_results(input_path):
    img = Image.open(input_path).convert('L')
    prewitt_result = prewitt_edge_detection(input_path)
    canny_result = canny_edge_detection(input_path)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Oryginalny")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Prewitt")
    plt.imshow(prewitt_result, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Canny")
    plt.imshow(canny_result, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_image_path = r"C:\Users\antek\PycharmProjects\PythonProject3\piramidalne\pg.jpg"
    display_results(input_image_path)
