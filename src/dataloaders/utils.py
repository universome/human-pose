def batch_to(batch, *args, **kwargs):
    images, targets = batch
    images = [img.to(*args, **kwargs) for img in images]

    for target in targets:
        target['labels'] = target['labels'].to(*args, **kwargs)
        target['boxes'] = target['boxes'].to(*args, **kwargs)
        target['masks'] = target['masks'].to(*args, **kwargs)

    return images, targets
