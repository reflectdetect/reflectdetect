from matplotlib import pyplot as plt
from robotpy_apriltag import AprilTagDetection


def show_panel(img, tags: list[AprilTagDetection], corners: list[float], output_path: str | None = None):
    fig_2d = plt.figure()
    ax = fig_2d.subplots(1, 1)
    ax.imshow(img, cmap="grey")

    for tag in tags:
        ax.scatter(tag.getCenter().x, tag.getCenter().y)
    x, y = zip(*corners)

    # Append the first point to the end to close the rectangle/polygon
    x = list(x) + [x[0]]
    y = list(y) + [y[0]]
    ax.plot(x, y, linewidth=1)

    if output_path is not None:
        fig_2d.savefig(output_path)
    else:
        fig_2d.show()
    plt.close(fig_2d)
