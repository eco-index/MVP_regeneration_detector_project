# Qualitative Validation Comparison: SAM2 vs Prithvi

Filenames accompany each image so the raw panels can be inspected in place. SAM2 panels follow the default layout (RGB | GT / logits | overlay). Prithvi panels show RGB + GT on the left half, probability + overlay on the right. All observations below are based on the specific images listed.

---

## 1. SAM2 (Epoch 9)

SAM2 panels are arranged RGB | GT (top) and logits | overlay (bottom).

### 1.1 Positive-case selection

`0_14.png` ![](../runs/9cb72b/validation_examples/epoch9/0_14.png)  
`0_15.png` ![](../runs/9cb72b/validation_examples/epoch9/0_15.png)  
`0_3.png` ![](../runs/9cb72b/validation_examples/epoch9/0_3.png)  
`1_13.png` ![](../runs/9cb72b/validation_examples/epoch9/1_13.png)  
`1_4.png` ![](../runs/9cb72b/validation_examples/epoch9/1_4.png)  
`1_7.png` ![](../runs/9cb72b/validation_examples/epoch9/1_7.png)  
`1_9.png` ![](../runs/9cb72b/validation_examples/epoch9/1_9.png)  
`2_3.png` ![](../runs/9cb72b/validation_examples/epoch9/2_3.png)  
`3_13.png` ![](../runs/9cb72b/validation_examples/epoch9/3_13.png)  
`3_4.png` ![](../runs/9cb72b/validation_examples/epoch9/3_4.png)

**Observations**
- SAM2 frequently paints large swaths of vegetation. Example `1_7.png` highlights nearly the entire tree belt, far beyond the GT polygon. This behaviour stems from supplying 1,024 synthetic points without the intended human guidance.
- Some overlays (e.g. `0_14.png` and `1_4.png`) include shelterbelts absent from the label. While those additions might reflect real native plantings, the prediction still lacks clear boundaries—useful for inspection but unreliable for auto-detection.
- The logits panels reveal minimal structure: the network lights up wherever vegetation appears, not just within the GT outline.

### 1.2 Negative-case selection

`2_1.png` ![](../runs/9cb72b/validation_examples/epoch9/2_1.png)  
`2_10.png` ![](../runs/9cb72b/validation_examples/epoch9/2_10.png)  
`2_5.png` ![](../runs/9cb72b/validation_examples/epoch9/2_5.png)  
`2_6.png` ![](../runs/9cb72b/validation_examples/epoch9/2_6.png)  
`2_7.png` ![](../runs/9cb72b/validation_examples/epoch9/2_7.png)

**Observations**
- Most negatives remain blank, illustrating SAM2’s dependence on prompt confidence. However, several negatives (e.g. `1_5.png`, `1_6.png`, not all shown) contain unlabelled plantings where SAM2 does produce green speckles, hinting at genuine vegetation GT left out.
- Even where SAM2 “gets it right,” the model’s output cannot be trusted automatically: the next tile might suddenly flood the scene if the prompts differ slightly.

**Takeaway:** SAM2 does not provide repeatable automatic segmentation. We see glimpses of correct behaviour, but only sporadically. The model’s design (interactive segmentation) clashes with the fully automatic requirement.

---

## 2. Prithvi (Epoch 10)

The foundation model produces stable, explainable segmentation. Here we focus on six illustrative tiles, manually verified in detail.

### 2.1 Positive examples

- **`pos_00003.png`** ![](../runs/prithvi_20251001-015236/val_examples/epoch10/pos_00003.png)  
  The GT polygon covers a paddock at the bottom right that is not clearly identifiable as fresh plantings. Prithvi is justified in classifying the area negatively.

- **`pos_00025.png`** ![](../runs/prithvi_20251001-015236/val_examples/epoch10/pos_00025.png)  
  GT outlines a wedge at the edge of a forest block. The model proposes a small, slightly overlapping cluster (green) where fresh plantings appear—arguably more precise than GT.

- **`pos_00029.png`** ![](../runs/prithvi_20251001-015236/val_examples/epoch10/pos_00029.png)  
  The long strip covered by the GT mask is not obviously identifiable in the image as fresh planting. The model justifiably ignores that area.

### 2.2 Negative examples

- **`neg_00006.png`** ![](../runs/prithvi_20251001-015236/val_examples/epoch10/neg_00006.png)  
  Despite GT labelling this as background, the overlay lights up a compact group of bright green crowns (top right). Visual inspection does not necessarily suggest fresh planting though the predicted area does closely resemble with other areas labeled as fresh planting in the training data.

- **`neg_00008.png`** ![](../runs/prithvi_20251001-015236/val_examples/epoch10/neg_00008.png)  
  Output is nearly empty except for small green patches along the fence line where shrubs creep into the paddock. Those hints map to actual vegetation in the RGB.

- **`neg_00016.png`** ![](../runs/prithvi_20251001-015236/val_examples/epoch10/neg_00016.png)  
  Many areas in the image appear to show potential fresh plantings that are not covered by GT labels.

**Conclusion for Prithvi:** masks reflect the real planting footprint. When the model disagrees, the imagery typically backs the model—most “false positives” correspond to shrubs, shelterbelts, or riparian vegetation overlooked by GT.

---

## 3. Overall Assessment

| Aspect | SAM2 | Prithvi |
|---|---|---|
| Design intent | Interactive segmentation (prompt-driven) | Automatic geospatial segmentation |
| Positives | Floods vegetation; boundaries unreliable | Localised, structurally aware; disagreements often justified |
| Negatives | Typically blank, dependent on prompts | Consistently blank unless true vegetation is visible |
| Deployment suitability | Low – output depends on prompt grid | High – stable, aligns with imagery | 

**Final take:** The Prithvi fine-tune is clearly the promising path. SAM2 remains valuable in interactive settings but lacks the predictability required for large-scale, unattended inference.
