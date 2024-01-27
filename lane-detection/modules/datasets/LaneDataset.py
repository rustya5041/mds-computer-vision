from modules.DependentLibraries import *
    
class LaneDataset(torch.utils.data.Dataset):
    DEFAULT_IMSIZE = (512, 256)

    def __init__(self, dataset_path, train=True, size=DEFAULT_IMSIZE, transform=True):
        self.dataset_path = dataset_path
        self.train = train
        self.size = size
        self.transform = transform
        self.data = []
        
        if self.train: label = [os.path.join(self.dataset_path, f'label_data_{num}.json') for num in ['0313', '0601']]
        else: label = [os.path.join(self.dataset_path, 'label_data_0531.json')]

        for file in label:
            with open(file) as f:
                for line in f:
                    info = json.loads(line)
                    image = info["raw_file"]
                    lanes = info["lanes"]
                    h_samples = info["h_samples"]
                    lanes_coords = []
                    for lane in lanes:
                        x = np.array([lane]).T
                        y = np.array([h_samples]).T
                        xy = np.hstack((x, y))
                        idx = np.where(xy[:, 0] > 0)
                        lane_coords = xy[idx]
                        lanes_coords.append(lane_coords)
                    self.data.append((image, lanes_coords))
    
    def create_img(self, h, w, lanes, image_type):
        image = np.zeros((h, w))
        for i, lane in enumerate(lanes):
            color = 1 if image_type == "segmentation" else i + 1
            cv2.polylines(image, [lane], False, color, 10)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_NEAREST)

        return image
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_path, self.data[idx][0])
        image = cv2.imread(image_path)
     
        h, w, c = image.shape
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lanes = self.data[idx][1]

        segmentation_image = self.create_img(h, w, lanes, "segmentation")
        instance_image = self.create_img(h, w, lanes, "instance")
        
        if self.train:
            image = self.transform(image)
            segmentation_image = self.transform(segmentation_image)
            instance_image = self.transform(instance_image)
    
        image = image[..., None]
        instance_image = instance_image[..., None]
        
        image = torch.from_numpy(image).float().permute((2, 0, 1))
        segmentation_image = torch.from_numpy(segmentation_image.copy())
        instance_image =  torch.from_numpy(instance_image.copy()).permute((2, 0, 1))
        segmentation_image = segmentation_image.to(torch.int64)
        
        return image, segmentation_image, instance_image

    def __len__(self):
        return len(self.data)