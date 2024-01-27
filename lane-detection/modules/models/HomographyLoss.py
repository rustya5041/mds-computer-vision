from modules.DependentLibraries import *

class HomographyLoss(torch.nn.modules.loss._Loss):
    def __init__(self, order=2, device='cuda'):
        super(HomographyLoss, self).__init__()
        self._device = device
        self._order = order
        self._R = torch.tensor(
            [[-2.0484e-01, -1.7122e+01,  3.7991e+02],
             [ 0.0000e+00, -1.6969e+01,  3.7068e+02],
             [ 0.0000e+00, -4.6739e-02,  1.0000e+00]],
            dtype=torch.float32, device='cuda')

    def forward(self, inputs, targets):
        losses = []
        for correction, lanes_coords in zip(inputs, targets):
            for lane in lanes_coords:
                if lane.sum() == 0: continue
                
                lane_loss = self.comp_lane_loss(correction, lane)
                
                if lane_loss is None: continue

                losses.append(lane_loss)
            mean_loss = torch.stack(losses).mean()
        return mean_loss
    
    def comp_lane_loss(self, correction, lane):
        H_correction_indices = [0, 1, 2, 4, 5, 7]
        H_correction = correction.flatten()
        H = self._R.flatten().clone()
        H[H_correction_indices] = H_correction
        H = H.reshape((3, 3))
    
        points = lane.T
        index = points.nonzero(as_tuple=True)[0].max().item() + 1
        points = points[:index, :]
        ones_col = torch.ones(points.shape[0], device='cuda')
        P = torch.column_stack((points, ones_col)).T.to(torch.float32)
        P_transformed = H @ P
    
        x_transformed = P_transformed[0, :].T
        y_transformed = P_transformed[1, :]
        Y = torch.column_stack((y_transformed ** 2, y_transformed, ones_col))
        w = torch.linalg.solve(Y.T @ Y, Y.T @ x_transformed)
    
        x_predicted = Y @ w
        
        P_predicted = torch.column_stack((x_predicted, y_transformed, ones_col)).T
        P_reprojected = torch.linalg.inv(H.detach()) @ P_predicted
    
        P_reprojected = P_reprojected / P_reprojected[2, :]
        loss = ((points[:, 0].T - P_reprojected[0, :]) ** 2).mean()
        return loss        