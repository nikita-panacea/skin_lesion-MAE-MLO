"""
Test script to verify model dimensions and architecture
"""
import torch
from models.mae_hybrid import MAEHybrid
from models.hybrid_classifier import HybridClassifier
from models.masking_module import UNetMaskingModule


def test_mae():
    """Test MAE forward pass"""
    print("\n" + "="*60)
    print("Testing MAE Model")
    print("="*60)
    
    batch_size = 2
    img_size = 224
    patch_size = 16
    mask_ratio = 0.75
    
    # Create model
    mae = MAEHybrid(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        encoder_dim=768,
        decoder_dim=512,
        decoder_depth=4,
        norm_pix_loss=True
    )
    
    # Create dummy input
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Input shape: {images.shape}")
    
    # Embed patches
    patches = mae.patch_embed(images)
    print(f"Patches shape: {patches.shape}")
    
    # Create dummy masking
    num_patches = patches.shape[1]
    len_keep = int(num_patches * (1 - mask_ratio))
    
    # Random masking for test
    noise = torch.rand(batch_size, num_patches)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    
    x_masked = torch.gather(
        patches, dim=1, 
        index=ids_keep.unsqueeze(-1).repeat(1, 1, patches.shape[2])
    )
    
    mask = torch.ones(batch_size, num_patches)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    mask_prob = noise
    
    print(f"Masked patches shape: {x_masked.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Forward pass
    try:
        pred = mae(x_masked, mask, ids_restore, mask_prob)
        print(f"Prediction shape: {pred.shape}")
        
        # Test reconstruction loss
        target = mae.patchify(images)
        print(f"Target shape: {target.shape}")
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        print(f"Loss: {loss.item():.4f}")
        
        print("✓ MAE test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ MAE test FAILED: {e}")
        return False


def test_classifier():
    """Test classifier forward pass"""
    print("\n" + "="*60)
    print("Testing Classifier Model")
    print("="*60)
    
    batch_size = 2
    img_size = 224
    num_classes = 8
    
    # Create model
    classifier = HybridClassifier(
        num_classes=num_classes,
        pretrained_mae=None,
        freeze_encoder=False
    )
    
    # Create dummy input
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Input shape: {images.shape}")
    
    try:
        logits = classifier(images)
        print(f"Logits shape: {logits.shape}")
        print(f"Expected shape: [{batch_size}, {num_classes}]")
        
        assert logits.shape == (batch_size, num_classes), \
            f"Shape mismatch: got {logits.shape}, expected ({batch_size}, {num_classes})"
        
        # Test feature extraction
        features = classifier.extract_features(images)
        print(f"Features shape: {features.shape}")
        
        print("✓ Classifier test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Classifier test FAILED: {e}")
        return False


def test_masking():
    """Test masking module"""
    print("\n" + "="*60)
    print("Testing Masking Module")
    print("="*60)
    
    batch_size = 2
    img_size = 224
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2
    embed_dim = 768
    mask_ratio = 0.75
    
    # Create model (without pretrained UNet for testing)
    masking = UNetMaskingModule(
        pretrained_unet_path=None,
        num_patches=num_patches,
        embed_dim=embed_dim,
        learnable=True
    )
    
    # Create dummy inputs
    images = torch.randn(batch_size, 3, img_size, img_size)
    patches = torch.randn(batch_size, num_patches, embed_dim)
    
    print(f"Images shape: {images.shape}")
    print(f"Patches shape: {patches.shape}")
    
    try:
        x_masked, mask, ids_restore, mask_prob = masking(
            images, patches, mask_ratio=mask_ratio, random=False
        )
        
        print(f"Masked patches shape: {x_masked.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Ids restore shape: {ids_restore.shape}")
        print(f"Mask prob shape: {mask_prob.shape}")
        
        # Verify shapes
        len_keep = int(num_patches * (1 - mask_ratio))
        assert x_masked.shape == (batch_size, len_keep, embed_dim), \
            f"Masked patches shape mismatch"
        assert mask.shape == (batch_size, num_patches), \
            f"Mask shape mismatch"
        
        print("✓ Masking test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Masking test FAILED: {e}")
        return False


def test_end_to_end():
    """Test complete pipeline"""
    print("\n" + "="*60)
    print("Testing End-to-End Pipeline")
    print("="*60)
    
    batch_size = 2
    img_size = 224
    patch_size = 16
    num_classes = 8
    mask_ratio = 0.75
    
    # Create models
    mae = MAEHybrid(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        encoder_dim=768,
        decoder_dim=512,
        decoder_depth=4,
        norm_pix_loss=True
    )
    
    classifier = HybridClassifier(
        num_classes=num_classes,
        pretrained_mae=None,
        freeze_encoder=False
    )
    
    masking = UNetMaskingModule(
        pretrained_unet_path=None,
        num_patches=(img_size // patch_size) ** 2,
        embed_dim=768,
        learnable=True
    )
    
    # Create dummy data
    images = torch.randn(batch_size, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    try:
        # Step 1: MAE reconstruction
        patches = mae.patch_embed(images)
        x_masked, mask, ids_restore, mask_prob = masking(
            images, patches, mask_ratio=mask_ratio, random=False
        )
        pred = mae(x_masked, mask, ids_restore, mask_prob)
        
        print(f"Step 1 - Reconstruction: {pred.shape}")
        
        # Step 2: Classification on real images
        logits_real = classifier(images)
        print(f"Step 2 - Real classification: {logits_real.shape}")
        
        # Step 3: Classification on reconstructed images
        fake_images = mae.unpatchify(pred)
        fake_images = torch.clamp(fake_images, 0, 1)
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        fake_images = (fake_images - mean) / std
        
        logits_fake = classifier(fake_images)
        print(f"Step 3 - Fake classification: {logits_fake.shape}")
        
        print("✓ End-to-end test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ End-to-end test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MODEL DIMENSION VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("MAE", test_mae()))
    results.append(("Classifier", test_classifier()))
    results.append(("Masking", test_masking()))
    results.append(("End-to-End", test_end_to_end()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)