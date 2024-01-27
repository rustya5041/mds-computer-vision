from modules.DependentLibraries import *
from modules.models.DiscriminativeLoss import DiscriminativeLoss

def define_labels(self):
    if self.train: label = [os.path.join(self.dataset_path, f'label_data_{num}.json') for num in ['0313', '0601']]
    else: label = [os.path.join(self.dataset_path, 'label_data_0531.json')]
    return label


def load_labels(path, dump_list):
    "This function loads the labels from the given path and returns a list of labels"
    with open(path, 'r') as f:
        for line in f:
            if line.strip() not in dump_list:
                dump_list.append(line.strip())
    return dump_list


def create_img(h, w, l):
    "This function creates an image with the given lane points"
    image = np.zeros((h, w))
    color = 10
    cv2.polylines(image, l, False, color, thickness=15)
    return image


def logger(log_directory_name):
    """
    This function creates a directory and returns two SummaryWriter objects
    for training and testing logging.
    """
    dir = log_directory_name

    train_dir, test_dir = [f'{dir}/{i}' for i in ['logs_train', 'logs_val']]

    os.makedirs(dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    writer_train = SummaryWriter(log_dir=train_dir)
    writer_val = SummaryWriter(log_dir=test_dir)
    return writer_train, writer_val


def visualize_lane_dataset(dataloader):
    image, segmentation_image, instance_image = next(iter(dataloader))

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(image[0, 0], cmap='gray')
    plt.title('Image')

    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_image[0], cmap='binary')
    plt.title('Segmentation')
    
    plt.subplot(1, 3, 3)
    plt.imshow(instance_image[0, 0], cmap='turbo')
    plt.title('Instance')
    plt.show()
    return None


def plot_segmentation(enet, val_dataloader, device):
    images, binary_labels, instance_labels = next(iter(val_dataloader))
    images = images.to(device)/255.
    
    assert enet is not None, "Please load the model first!"
    enet.eval()

    with torch.no_grad():
        binary_logits, instance_final_logits = enet(images)

    for e in range(len(images)):
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(images[e].cpu().permute((1, 2, 0)), cmap='gray')
        plt.title('Init image')

        plt.subplot(1, 3, 2)
        plt.imshow(binary_logits[e].cpu().numpy().squeeze()[0], cmap='binary')
        plt.title('Binary segmentation')
        
        plt.subplot(1, 3, 3)
        plt.imshow(instance_final_logits[e].cpu().numpy().squeeze()[4], cmap='twilight_shifted')
        plt.title('Instance segmentation')
        plt.show()
    return None

def compute_enet_loss(binary_output, instance_output, binary_label, instance_label):
    """
    Computes the loss for ENet model.
        :param binary_output: torch.tensor, shape: (batch_size, num_classes, H, W)
        :param instance_output: torch.tensor, shape: (batch_size, embedding_dim, H, W)
        :param binary_label: torch.tensor, shape: (batch_size, H, W)
        :param instance_label: torch.tensor, shape: (batch_size, H, W)
    returns: binary_loss, instance_loss
    """
    ce_loss = nn.CrossEntropyLoss()
    binary_loss = ce_loss(binary_output, binary_label)

    ds_loss = DiscriminativeLoss(delta_var=0.5, delta_dist=3, alpha=1.0, beta=1.0, gamma=0.001, device="cuda")
    instance_loss = ds_loss(instance_output, instance_label)
    return binary_loss, instance_loss

def train_enet(enet_model, tr_dataloader, val_dataloader, optimizer, device, epochs=50):
    """
    Training loop for ENet.
        param epochs: number of epochs to train for
        returns: None
    """
    writer_train, writer_val = logger('logs')  # init logger for tensorboard

    binary_losses_epoch = []
    instance_losses_epoch = []
    train_accuracies = []

    valid_accuracies = []
    binary_losses_epoch_val = []
    instance_losses_epoch_val = []


    for epoch in range(1, epochs+1):

        print(f"Epoch {epoch}:")
        
        # training loop
        enet_model.train()
        losses_tr = []
        correct_binary = 0
        total_pixels = 0
        
        for batch in tqdm(tr_dataloader):
            img, binary_target, instance_target = batch
            img = img.to(device)/255.
            binary_target = binary_target.to(device)
            instance_target = instance_target.to(device)

            optimizer.zero_grad()

            binary_logits, instance_emb = enet_model(img)

            binary_loss, instance_loss = compute_enet_loss(binary_logits, instance_emb, binary_target, instance_target)
            loss = binary_loss + instance_loss
            loss.backward()

            optimizer.step()

            losses_tr.append((binary_loss.detach().cpu(), instance_loss.detach().cpu()))

            binary_preds = torch.argmax(binary_logits, dim=1)
            correct_binary += torch.sum(binary_preds == binary_target).item()
            total_pixels += binary_target.numel()

        binary_accuracy = correct_binary / total_pixels
        train_accuracies.append(binary_accuracy)

        mean_losses_tr = np.array(losses_tr).mean(axis=0)
        binary_losses_epoch.append(mean_losses_tr[0])
        instance_losses_epoch.append(mean_losses_tr[1])

        # Log metrics to TensorBoard
        writer_train.add_scalar("Binary Loss", mean_losses_tr[0], epoch)
        writer_train.add_scalar("Instance Loss", mean_losses_tr[1], epoch)
        writer_train.add_scalar("Binary Accuracy", binary_accuracy, epoch)

        # Log details of all layers in histogram format
        for name, param in enet_model.named_parameters():
            writer_train.add_histogram(name, param.clone().cpu().data.numpy(), global_step=epoch)

        # validation loop
        enet_model.eval()
        losses_val = []
        correct_binary = 0
        total_pixels = 0

        for batch in tqdm(val_dataloader):
            img, binary_target, instance_target = batch
            img = img.to(device)/255.
            binary_target = binary_target.to(device)
            instance_target = instance_target.to(device)

            binary_logits, instance_emb = enet_model(img)

            binary_loss, instance_loss = compute_enet_loss(binary_logits, instance_emb, binary_target, instance_target)

            losses_val.append((binary_loss.detach().cpu(), instance_loss.detach().cpu()))

            binary_preds = torch.argmax(binary_logits, dim=1)
            correct_binary += torch.sum(binary_preds == binary_target).item()
            total_pixels += binary_target.numel()

        binary_accuracy_val = correct_binary / total_pixels
        valid_accuracies.append(binary_accuracy_val)

        mean_losses_val = np.array(losses_val).mean(axis=0)
        binary_losses_epoch_val.append(mean_losses_val[0])
        instance_losses_epoch_val.append(mean_losses_val[1])

        # Log metrics to TensorBoard
        writer_val.add_scalar("Binary Loss", mean_losses_val[0], epoch)
        writer_val.add_scalar("Instance Loss", mean_losses_val[1], epoch)
        writer_val.add_scalar("Binary Accuracy", binary_accuracy_val, epoch)


        # Print and save results for this epoch
        print(f"Binary Loss train = {mean_losses_tr[0]:.4f}, Instance Loss train = {mean_losses_tr[1]:.4f}, Binary Accuracy train = {binary_accuracy:.4f}\nBinary Loss val = {mean_losses_val[0]:.4f}, Instance Loss val = {mean_losses_val[1]:.4f}, Binary Accuracy val = {binary_accuracy_val:.4f}\n\n")

    writer_train.close()
    writer_val.close()

def calc_enet_segmentation_scores(enet_model, dataset, device): 
    # print accuracy, loss and IoU on validation set
    val_dataset = dataset
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

    enet_model.eval()

    correct_binary = 0
    total_pixels = 0
    binary_losses = []
    instance_losses = []

    for batch in tqdm(val_dataloader):
        img, binary_target, instance_target = batch
        img = img.to(device)/255.
        binary_target = binary_target.to(device)
        instance_target = instance_target.to(device)
    
        binary_logits, instance_emb = enet_model(img)

        assert compute_enet_loss is not None, "compute_loss not defined/implemented"
        binary_loss, instance_loss = compute_enet_loss(binary_logits, instance_emb, binary_target, instance_target)
    
        binary_losses.append(binary_loss.detach().cpu())
        instance_losses.append(instance_loss.detach().cpu())
    
        binary_preds = torch.argmax(binary_logits, dim=1)
        correct_binary += torch.sum(binary_preds == binary_target).item()
        total_pixels += binary_target.numel()

    binary_accuracy = correct_binary / total_pixels
    mean_binary_loss = np.array(binary_losses).mean()
    mean_instance_loss = np.array(instance_losses).mean()

    print(f"Binary Loss = {mean_binary_loss:.3f}, Instance Loss = {mean_instance_loss:.3f}, Binary Accuracy = {binary_accuracy:.3f}\n\n")
    return mean_binary_loss, mean_instance_loss, binary_accuracy

def train_hnet(hnet, dataloader, epochs, crit, device, optimizer):
    for epoch in range(1, epochs+1):
        hnet.train()
        losses = []
        for inputs, points in tqdm(dataloader, leave=False):
            inputs = inputs.to(device)
            points = points.to(device)    

            coefs = hnet(inputs)
            loss = crit(coefs, points)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())

        epoch_loss = np.array(losses).mean()

        print(f"Epoch {epoch}, loss = {epoch_loss}")