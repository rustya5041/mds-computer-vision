from modules.DependentLibraries import *

class HomographyPredictionDataset(torch.utils.data.Dataset):
    HNET_DEFAULT_IMSIZE = (128, 64)
    def __init__(self, path="./TUSimple/train_set", train=True, size=HNET_DEFAULT_IMSIZE):
        self.dataset_path = path
        self.train = train
        self.size = size
        self.max_lanes = 0
        self.max_points = 0
        self.image_list = []
        self.lanes_list = []
        labels = []

        if self.train: labels = [os.path.join(self.dataset_path, f'label_data_{num}.json') for num in ['0313', '0601']]
        else: labels = [os.path.join(self.dataset_path, 'label_data_0531.json')]
            

        for file in labels:
            for line in open(file).readlines():
                info_dict = json.loads(line)
                self.image_list.append(info_dict['raw_file'])
                h_samples = info_dict['h_samples']
                lanes = info_dict['lanes']
                self.max_lanes = max(self.max_lanes, len(lanes))
                xy_list = []

                for lane in lanes:
                    y = np.array([h_samples], dtype=np.float64).T
                    x = np.array([lane], dtype=np.float64).T
                    xy = np.hstack((x, y))
                    index = np.where(xy[:, 0] > 2)
                    xy_list.append(xy[index])
                    self.max_points = max(self.max_points, len(xy[index]))
                self.lanes_list.append(xy_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.image_list[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        original_height, original_width, _ = image.shape
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)

        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float() / 255

        # creating output with shape [max_lanes, 2, max_points]
        buffer = None

        for lane in self.lanes_list[idx]:
            lane = np.expand_dims(np.pad(np.swapaxes(lane, 0, 1),
                                         pad_width=((0, 0), (0, self.max_points - lane.shape[0])),
                                         mode='constant',
                                         constant_values=0), 0)

            if buffer is not None:
                buffer = np.concatenate((buffer, lane), 0)
            else:
                buffer = lane
        
        buffer[:, 0, :] = buffer[:, 0, :] / original_width
        buffer[:, 1, :] = buffer[:, 1, :] / original_height
        ground_truth_trajectory = torch.from_numpy(np.pad(buffer,
                                                          pad_width=(
                                                              (0, self.max_lanes - buffer.shape[0]),(0, 0),(0, 0)),
                                                          mode='constant', constant_values = 0))

        return image, ground_truth_trajectory

    def __len__(self):
        return len(self.image_list)