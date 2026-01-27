"""
Test script to verify gradient flow in MLO-MAE
Tests all components individually and together
"""
import torch
import torch.nn as nn
import sys
sys.path.append('.')

from models.mae_hybrid import MAEHybrid
from models.hybrid_classifier import HybridClassifier
from models.masking_module import UNetMaskingModule


def test_mae():
    """Test MAE model"""
    print("\n" + "="*60)
    print("Testing MAE Model")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mae = MAEHybrid(
        img_size=224,
        patch_size=16,
        encoder_dim=768,
        decoder_dim=512,
        decoder_depth=4
    ).to(device)
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Embed patches
    patches = mae.patch_embed(images)
    print(f"Input shape: {images.shape}")
    print(f"Patches shape: {patches.shape}")
    
    # Simulate masking
    mask_ratio = 0.75
    N = patches.shape[1]
    len_keep = int(N * (1 - mask_ratio))
    
    # Random mask
    noise = torch.rand(batch_size, N, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    
    x_masked = torch.gather(
        patches, 
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, patches.shape[2])
    )
    
    mask = torch.ones(batch_size, N, device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    print(f"Masked patches shape: {x_masked.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Forward pass
    pred = mae(x_masked, mask, ids_restore)
    print(f"Prediction shape: {pred.shape}")
    
    # Compute loss
    target = mae.patchify(images)
    print(f"Target shape: {target.shape}")
    
    loss = mae.forward_loss(images, pred, mask)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in mae.parameters() if p.requires_grad)
    
    print(f"✓ MAE test {'PASSED' if has_grad else 'FAILED'}")
    return has_grad


def test_classifier():
    """Test classifier model"""
    print("\n" + "="*60)
    print("Testing Classifier Model")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = HybridClassifier(
        num_classes=8,
        pretrained_mae=None
    ).to(device)
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, 8, (batch_size,)).to(device)
    
    print(f"Input shape: {images.shape}")
    
    # Forward pass
    logits = classifier(images)
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, 8]")
    
    # Compute loss
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in classifier.parameters() if p.requires_grad)
    
    # Test feature extraction
    features = classifier.extract_features(images)
    print(f"Features shape: {features.shape}")
    
    print(f"✓ Classifier test {'PASSED' if has_grad else 'FAILED'}")
    return has_grad


def test_masking():
    """Test masking module"""
    print("\n" + "="*60)
    print("Testing Masking Module")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masking = UNetMaskingModule(
        pretrained_unet_path=None,  # No UNet for testing
        num_patches=196,
        embed_dim=768,
        learnable=True
    ).to(device)
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    patches = torch.randn(batch_size, 196, 768).to(device)
    
    print(f"Images shape: {images.shape}")
    print(f"Patches shape: {patches.shape}")
    
    # Forward pass
    x_masked, mask, ids_restore, mask_prob = masking(
        images, patches, mask_ratio=0.75, random=False
    )
    
    print(f"Masked patches shape: {x_masked.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Ids restore shape: {ids_restore.shape}")
    print(f"Mask prob shape: {mask_prob.shape}")
    
    # Check gradient flow through learned part
    if masking.learnable:
        # Compute dummy loss on mask_prob
        loss = mask_prob.mean()
        loss.backward()
        
        # Check if refinement network has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in masking.mask_refine.parameters())
    else:
        has_grad = True  # No learnable params to check
    
    print(f"✓ Masking test {'PASSED' if has_grad else 'FAILED'}")
    return has_grad


def test_end_to_end():
    """Test end-to-end integration"""
    print("\n" + "="*60)
    print("Testing End-to-End Pipeline")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create all models
    mae = MAEHybrid(img_size=224, patch_size=16, encoder_dim=768).to(device)
    classifier = HybridClassifier(num_classes=8).to(device)
    masking = UNetMaskingModule(
        pretrained_unet_path=None,
        num_patches=196,
        embed_dim=768,
        learnable=True
    ).to(device)
    
    # Create dummy data
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    labels = torch.randint(0, 8, (batch_size,)).to(device)
    
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Step 1: MAE reconstruction
    patches = mae.patch_embed(images)
    x_masked, mask, ids_restore, mask_prob = masking(
        images, patches, mask_ratio=0.75
    )
    pred = mae(x_masked, mask, ids_restore)
    loss_mae = mae.forward_loss(images, pred, mask, mask_prob)
    
    print(f"Step 1 - Reconstruction: {pred.shape}")
    
    # Step 2: Classification on real images
    logits_real = classifier(images)
    loss_cls_real = nn.CrossEntropyLoss()(logits_real, labels)
    
    print(f"Step 2 - Real classification: {logits_real.shape}")
    
    # Step 3: Classification on fake images
    fake_images = mae.unpatchify(pred).detach()
    fake_images = torch.clamp(fake_images, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    fake_images = (fake_images - mean) / std
    
    logits_fake = classifier(fake_images)
    loss_cls_fake = nn.CrossEntropyLoss()(logits_fake, labels)
    
    print(f"Step 3 - Fake classification: {logits_fake.shape}")
    
    # Combined loss
    loss_total = loss_mae + loss_cls_real + 0.5 * loss_cls_fake
    
    # Backward pass
    loss_total.backward()
    
    # Check gradients in all models
    mae_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in mae.parameters() if p.requires_grad)
    cls_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in classifier.parameters() if p.requires_grad)
    mask_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                    for p in masking.parameters() if p.requires_grad)
    
    all_pass = mae_grad and cls_grad and mask_grad
    
    print(f"✓ End-to-end test {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MODEL DIMENSION VERIFICATION")
    print("="*60)
    
    tests = {
        'MAE': test_mae,
        'Classifier': test_classifier,
        'Masking': test_masking,
        'End-to-End': test_end_to_end
    }
    
    results = {}
    for name, test_fn in tests.items():
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"✗ {name} test FAILED with error: {e}")
            results[name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    print("\n" + "="*60)
    if all(results.values()):
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60 + "\n")
    
    return all(results.values())


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)