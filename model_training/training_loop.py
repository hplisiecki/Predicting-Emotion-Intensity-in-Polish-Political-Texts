from tqdm import tqdm
import torch
import numpy as np
import wandb


def training_loop(model, optimizer, scheduler, epochs, train_dataloader,
          val_dataloader, criterion, device, save_dir,
                  use_wandb = False):

    best_corr_total = 0
    for epoch_num in range(epochs):
        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = torch.stack(train_label, dim=0).to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            outputs = model(input_id, mask)

            # concatenate
            output_a = torch.cat(outputs, dim=0)
            del outputs

            batch_loss = criterion(output_a.float(), train_label.view(-1,1).float())
            del output_a, train_label

            total_loss_train += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss_val = 0
        correlations = [0 for _ in range(len(train_dataloader.dataset[0][-1]))]
        mean_preds = [0 for _ in range(len(train_dataloader.dataset[0][-1]))]

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = torch.stack(val_label, dim=0).to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                outputs = model(input_id, mask)

                val_output_a = torch.cat(outputs, dim=0)

                batch_loss = criterion(val_output_a.float(), val_label.view(-1,1).float())
                del val_output_a

                total_loss_val += batch_loss.item()

                outputs_detached = [o.cpu().detach().view(-1).numpy() for o in outputs]
                labels_np = [l.cpu().numpy() for l in val_label]

                for i in range(len(outputs_detached)):
                    correlations[i] += np.corrcoef(outputs_detached[i], labels_np[i])[0,1]
                    mean_preds[i] += np.mean(outputs_detached[i])


            total_corr = sum(correlations) /  len(outputs_detached)
            total_mean_preds = sum(mean_preds) / len(outputs_detached)

            if best_corr_total < total_corr / len(val_dataloader):
                best_corr_total = total_corr / len(val_dataloader)
                torch.save(model.state_dict(), save_dir)
                print('Saved model')

        if use_wandb:
            wandb_dict = {"loss": total_loss_train / len(train_dataloader), "lr": scheduler.get_last_lr()[0],
                          "epoch": epoch_num, "val_loss": total_loss_val/ len(val_dataloader)}
            for i, corr in enumerate(correlations):
                wandb_dict[f"corr_variable_{i}"] = corr / len(val_dataloader)
            wandb_dict["total_corr"] = total_corr / len(val_dataloader)
            for i, corr in enumerate(correlations):
                wandb_dict[f"mean_pred_{i}"] = mean_preds[i] / len(val_dataloader)
            wandb_dict["total_mean_pred"] = total_mean_preds / len(val_dataloader)

            if epoch_num % 2 == 0:
                wandb.log(wandb_dict)

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader): .10f} \
                | Val Loss: {total_loss_val / len(val_dataloader): .10f} | Val Corr: {total_corr / len(val_dataloader): .10f} \
                | Val Mean Pred: {total_mean_preds / len(val_dataloader): .10f}')
