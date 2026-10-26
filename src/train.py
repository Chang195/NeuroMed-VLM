from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

def train_one_epoch(
    image_encoder,
    text_encoder,
    image_projection,
    text_projection,
    loader,
    optimizer,
    temperature
):
    epoch_loss_sum = 0.0
    epoch_samples = 0
    progress_bar = tqdm(loader, desc="Training")

    # 下一步在这里实现Training Loop
    for step, batch in enumerate(progress_bar):
    
            images = batch["pixel_values"]
            input_ids = batch["input_ids"]
    
            assert images.shape[1:] == (3, 224, 224)
            assert images.shape[0] == input_ids.shape[0]
            assert input_ids.shape[1] <= 128
    
            with torch.no_grad():
                outputs = image_encoder(pixel_values=batch["pixel_values"])
            hidden = outputs.last_hidden_state
    
            cls_embedding = hidden[:, 0, :]
    
            with torch.no_grad():
                text_outputs = text_encoder(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"]
                                )
            hidden = text_outputs.last_hidden_state
    
            mask = batch["attention_mask"]
            mask_expanded = mask.unsqueeze(-1)
            masked_hidden = hidden * mask_expanded
            sum_hidden = masked_hidden.sum(dim=1)
            token_counts = mask_expanded.sum(dim=1)
            text_mean = sum_hidden / token_counts.clamp(min=1)
    
            image_features = cls_embedding
            text_features = text_mean
    
            image_projected = image_projection(image_features)
            text_projected = text_projection(text_features)
    
            image_norm = F.normalize(image_projected, p=2, dim=-1)
            text_norm = F.normalize(text_projected, p=2, dim=-1)
    
            similarity = image_norm @ text_norm.T
            logits = similarity / temperature
    
            labels = torch.arange(
                    images.shape[0],
                    device=logits.device
                    )
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2
    
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
    
            batch_size = batch["pixel_values"].shape[0]
            epoch_samples += batch_size 
            epoch_loss_sum += loss.item()*batch_size

            progress_bar.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            avg_loss=f"{epoch_loss_sum / epoch_samples:.4f}"
            )
        #    if step == 2500:
        #    break
    epoch_avg_loss = epoch_loss_sum / epoch_samples
    return epoch_avg_loss