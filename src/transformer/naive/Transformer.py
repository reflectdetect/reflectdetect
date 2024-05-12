from src.transformer.BaseTransformer import BaseTransformer
import numpy as np
import matplotlib.pyplot as plt

class Transformer(BaseTransformer):
    def transform(self, image_path, extraction_path) -> str:
        # Load extraction data
        # panel_id, ref, rad
        # calculate linear model ax+b based on ref rad
        # transform image


        x = [1, 2, 3, 4]
        y = [3, 5, 7, 10]  # 10, not 9, so the fit isn't perfect

        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)

        plt.plot(x, y, 'yo', x, poly1d_fn(x), '--k')
        plt.xlim(0, 5)
        plt.ylim(0, 12)
        return image_path
