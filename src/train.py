# Imports
import torch
import time 

def train_network(discriminator, generator, dataloader, num_epochs, criterion, optimiser_d, optimiser_g, fixed_noise, z_dim, device, checkpoint_dir='checkpoint'):
    """Train the model for a specified number of epochs, saving a checkpoint after each epoch, and generate images using fixed noise every 3 epochs."""
    # Training Loop
    epoch_image_history = [] # Save generated output images and corresponding epochs
    loss_history_iter, loss_history_epoch = [], [] # Track loss vs iterations and vs epochs
    iters = 0
    
    start_time = time.time() # Track total training time
    
    # For each epoch
    for epoch in range(num_epochs):
        epoch_start_time = time.time() # Track epoch time
        
        # For each batch in the dataloader
        for batch_idx, data in enumerate(dataloader):
            #############################################################
            # Update Discriminator: Maximize log(D(x)) + log(1 - D(G(z)))
            #############################################################
            discriminator.zero_grad()
        
            # Create labels for real and fake images
            real_images = data.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
        
            # Forward pass and compute loss with real images
            scores_d_real = discriminator(real_images).view(-1) # Reshape scores into 1D tensor
            loss_d_real = criterion(scores_d_real, real_labels)
            
            # Backward pass
            loss_d_real.backward()
            D_x = scores_d_real.mean().item() # D_x = average score (probability) assigned by discriminator to the real images
        
            # Generate fake images and forward pass through discriminator
            z = torch.randn(batch_size, z_dim, 1, 1).to(device) # Generate random noise in form of latent vector Z with shape (N, Z_DIM, 1, 1)
            fake_images = generator(z) # Generate fake images
            scores_d_fake = discriminator(fake_images.detach()).view(-1) # Forward pass with detached fake images to prevent gradient 
                                                                         # computation for the generator during forward pass of the 
                                                                         # discriminator as we only want to update the discriminator's 
                                                                         # weights in the first stage
            
            # Compute discriminator loss on the fake images
            loss_d_fake = criterion(scores_d_fake, fake_labels)
            loss_d_fake.backward()
            D_G_z1 = scores_d_fake.mean().item() # D_G_z1 = average score (probability) assigned by discriminator to the fake images (pre-updated D)
            
            # Sum discriminator loss over real and fake images
            loss_d = loss_d_real + loss_d_fake
    
            # Update discriminator
            optimiser_d.step()
            
            #########################################
            # Update Generator: Maximize log(D(G(z)))
            #########################################
            generator.zero_grad()
        
            # Forward pass with fake images through discriminator
            scores_g = discriminator(fake_images).view(-1)
    
            # Compute loss of generator using real labels (since our goal is to fool the discriminator)
            loss_g = criterion(scores_g, real_labels) 
            
            # Backward pass and optimisation for generator
            loss_g.backward()
            D_G_z2 = scores_g.mean().item() # D_G_z2 = average score (probability) assigned by Discriminator to the fake images (post-updated D)
            
            # Update generator
            optimiser_g.step()
            
            # Save loss for plotting by iteration, then increment iters
            loss_history_iter.append((iters+1, loss_d.item(), loss_g.item()))
            iters += 1
    
        # Save Generator's output on fixed_noise every 3 epochs
        if (epoch % 3 == 0) or ((epoch == num_epochs-1)):
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
            epoch_image_history.append((epoch+1, fake_images))
        
        # Track loss history for each epoch
        loss_history_epoch.append((epoch+1, loss_d.item(), loss_g.item()))
        
        # Save checkpoint after every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'discriminator_state_dict': discriminator.state_dict(),
            'optimiser_d_state_dict': optimiser_d.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'optimiser_g_state_dict': optimiser_g.state_dict(),
            'loss_history_iter': loss_history_iter,
            'loss_history_epoch': loss_history_epoch,
        }
        torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth.tar')
    
        # Output training stats at end of each epoch
        epoch_end_time = time.time()
        print('=' * 20)
        print(f'Epoch {epoch+1} | Loss_D = {loss_d.item():.4f} | Loss_G = {loss_g.item():.4f} | D_x = {D_x:.4f} | D_G_z1 = {D_G_z1:.4f} | D_G_z2 = {D_G_z2:.4f} | Time taken: {epoch_end_time - epoch_start_time:.2f} seconds')
        print('=' * 20)
        
    end_time = time.time()
    print(f"Total training time ({epoch+1} epochs): {end_time - start_time:.2f} seconds")
    
    return loss_history_iter, loss_history_epoch, epoch_image_history

def generate_test_images(generator, fixed_noise):
    """Generate test images using provided generator model and fixed noise."""
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    return fake_images