from utils import vis_utils
import csv

fieldnames = [
    'epoch', 'rmse', 'photo', 'mae', 'irmse', 'imae', 'mse', 'absrel', 'lg10', 'silog', 'squared_rel', 'delta1', 'delta2', 'delta3', 'data_time',
    'gpu_time', 'count'
]


class logger:

    def __init__(self, args):
        self.args = args
        if self.args.mode == 'val':
            self.csvfile_name = self.args.log_directory + "/result_test.csv"
        else:
            self.csvfile_name = self.args.log_directory + "/result_val.csv"
        self.best_csvfile_name = self.args.log_directory + "/result_best.csv"
        with open(self.csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(self.best_csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def _get_img_comparison_name(self, epoch, is_best=False):
        if self.args.mode == 'val':
            return self.args.log_directory + '/comparison_test.png'
        if self.args.mode == 'train':
            if is_best:
                return self.args.log_directory + '/comparison_best.png'
            else:
                return self.args.log_directory + '/comparison_' + str(epoch) + '.png'

    def conditional_save_img_comparison(self, i, data, pred, epoch, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
        # save 8 images for visualization
        skip = 100
        if i == 0:
            self.img_merge = vis_utils.merge_into_row(data, pred, predrgb, predg, extra, extra2, extrargb)
        elif i % skip == 0 and i < 8 * skip:
            row = vis_utils.merge_into_row(data, pred, predrgb, predg, extra, extra2, extrargb)
            self.img_merge = vis_utils.add_row(self.img_merge, row)
        elif i == 8 * skip:
            filename = self._get_img_comparison_name(epoch)
            vis_utils.save_image(self.img_merge, filename)

    def save_img_comparison_as_best(self, epoch):
        if self.args.mode == 'train':
            filename = self._get_img_comparison_name(epoch, is_best=True)
            vis_utils.save_image(self.img_merge, filename)

    def conditional_save_info(self, avg, epoch, is_best=False):
        if is_best:
            csvfile_name = self.best_csvfile_name
        else:
            csvfile_name = self.csvfile_name
        with open(csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'rmse': avg.sum_rmse,
                'photo': avg.sum_photometric,
                'mae': avg.sum_mae,
                'irmse': avg.sum_irmse,
                'imae': avg.sum_imae,
                'mse': avg.sum_mse,
                'silog': avg.sum_silog,
                'squared_rel': avg.sum_squared_rel,
                'absrel': avg.sum_absrel,
                'lg10': avg.sum_lg10,
                'delta1': avg.sum_delta1,
                'delta2': avg.sum_delta2,
                'delta3': avg.sum_delta3,
                'count': avg.count
            })
