from modules.DependentLibraries import *

class LaneDetector:
    DEFAULT_IMAGE_SIZE = (512, 256)
    
    def __init__(self, enet, hnet=None, device="cuda", with_projection=False):
        self._enet = enet
        self._hnet = hnet
        self._default_homography = torch.tensor(
            [[-2.0484e-01, -1.7122e+01,  3.7991e+02],
             [ 0.0000e+00, -1.6969e+01,  3.7068e+02],
             [ 0.0000e+00, -4.6739e-02,  1.0000e+00]],
            dtype=torch.float32, device=device)
        self._eps = 1.0
        self._device = device
        self._with_projection = with_projection
  
    def __call__(self, image, y_positions=None):
        image = self._preprocess_image(image)
        
        if y_positions is None:
            y_positions = np.linspace(50, image.shape[2], 30)

        binary_logits, instance_embeddings = self._enet(image)
        segmentation_map = binary_logits.argmax(dim=1)

        instances_map = self._cluster_img(segmentation_map, instance_embeddings)
        lanes = self._extract_lanes(instances_map)

        if self._with_projection:
            projected_lanes = self._project_lanes(lanes)
            coefs = self._fit(projected_lanes)

            y_positions_projected = self._project_y(y_positions)
            fitted_lanes = self._predict_lanes(coefs, y_positions_projected)

            reprojected_lanes = self._reproject(fitted_lanes)
            predicted_lanes = reprojected_lanes
        else:
            coefs = self._fit(lanes)
            fitted_lanes = self._predict_lanes(coefs, y_positions)
            predicted_lanes = fitted_lanes
        predicted_lanes = self._postprocess_result(predicted_lanes)
        res_inst, res_lanes = instances_map.cpu().numpy(), predicted_lanes.cpu().numpy()
        return res_inst, res_lanes
    
    def _cluster_img(self, segmentation_map, instance_embeddings):
        segmentation_map = segmentation_map.flatten()
        instance_embeddings = instance_embeddings.squeeze().permute(1, 2, 0).reshape(segmentation_map.shape[0], -1)
        
        mask_indices = segmentation_map.nonzero().flatten()
        cluster_data = instance_embeddings[mask_indices].detach().cpu()
        
        clusterer = DBSCAN(eps=self._eps)
        labels = clusterer.fit_predict(cluster_data)
        labels = torch.tensor(labels, dtype=instance_embeddings.dtype, device=self._device)
        
        instances_map = torch.zeros(instance_embeddings.shape[0], dtype=instance_embeddings.dtype, device=self._device)
        instances_map[mask_indices] = labels
        instances_map = instances_map.reshape(self.DEFAULT_IMAGE_SIZE[::-1])
        return instances_map
    
    def _extract_lanes(self, instances_map, scale=False):
        lanes = list()
        lane_indices = instances_map.unique()[1:]
        for index in lane_indices:
            coords = (instances_map == index).nonzero(as_tuple=True)
            if scale:
                coords = [c / 4 for c in coords]
            coords = coords[::-1]
            coords = torch.stack(coords).to(instances_map.dtype)
            lanes.append(coords)
        return lanes         
    
    def _fit(self, lanes):
        coefs = list()
        for lane in lanes:
            x = lane[0, :].unsqueeze(dim=1)
            y = lane[1, :]
            Y = torch.stack((y, torch.ones(y.shape[0], device=self._device))).T
            w = torch.linalg.solve(Y.T @ Y, Y.T @ x)
            coefs.append(w)
        return coefs

    def _postprocess_result(self, lanes):
        processed = list()
        for i, lane in enumerate(lanes):
            lane = lane.T
            lane[:, 2] = i
            ind1 = lane[:, 0] >= 0
            ind2 = lane[:, 0] <= 512
            index = torch.logical_and(ind1, ind2)
            lane = lane[index, :]
            processed.append(lane)
        return torch.cat(processed, dim=0)
    
    def _predict_lanes(self, coefs, y_positions):
        lanes = []
        
        for coef in coefs:
            c, d = coef
            lane = []
            for y in y_positions:
                x = c * y + d
                lane.append((x, y, 1))
            lanes.append(torch.tensor(lane, device=self._device).T)
        
        return lanes                
    
    def _preprocess_image(self, image):
        image = cv2.resize(image, self.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[..., None]
        image = torch.from_numpy(image).float().permute((2, 0, 1)).unsqueeze(dim=0).to(self._device)
        return image

    def _project_lanes(self, lanes):
        projected = list()
        for lane in lanes:
            ones = torch.ones((1, lane.shape[1]), device=self._device)
            P = torch.cat((lane, ones), dim=0)
            P_projected = self._default_homography @ P
            P_projected = P_projected / P_projected[2, :]
            projected.append(P_projected)
        return projected
    
    def _project_y(self, y_positions):
        y_positions = torch.from_numpy(y_positions).to(torch.float32).to(self._device)
        Y = torch.stack((torch.zeros(y_positions.shape[0], device=self._device), y_positions, torch.ones(y_positions.shape[0], device=self._device)))
        Y_projected = self._default_homography @ Y
        Y_projected = Y_projected / Y_projected[2, :]
        y_positions_projected = Y_projected[1, :]
        return y_positions_projected
    
    def _reproject(self, lanes):
        reprojected = list()
        for lane in lanes:
            lane_reprojected = torch.linalg.inv(self._default_homography) @ lane
            lane_reprojected = lane_reprojected / lane_reprojected[2, ]
            reprojected.append(lane_reprojected)
        return reprojected
    
    
    
