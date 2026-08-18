Failure Case: Repeated-Pattern Ambiguity

Observed error: 65.2 pixels

**1.Highly periodic search image**

  The search image contains several semiconductor regions with highly repetitive grid-like structures.
  Many local regions have nearly identical pitch, edge arrangement, and texture.
  
**2.Reference pattern is not globally unique**

  The reference image contains a small, repetitive FinFET/array-like structure.
  Similar structures occur at multiple locations within the search image.

**3.Model identified the correct structural family**

  The prediction is not a random location.
  It falls within a region containing a visually similar repeated pattern.
  This indicates that the learned feature extractor successfully recognized the relevant semiconductor structure.

**4.Wrong repeated instance selected**

  The model confused the true target site with a nearby visually similar repetition.
  Therefore, the main error is site disambiguation, rather than failure to recognize the pattern itself.

**5.Why pixel/feature matching becomes ambiguous**

  At the local scale, neighboring repetitions can have almost identical appearance.
  The feature-correlation response can therefore be strong at multiple candidate locations.
  Without a sufficiently distinctive surrounding context or an external positional prior, the model may select the wrong instance.

**6.Effect on coordinate prediction**

  The localization error is 65.2 px, even though the predicted point is visually close to the correct region.
  This illustrates why a small-looking visual displacement can correspond to a significant coordinate error in a high-precision inspection task.

**Key limitation**

  This is a fundamental limitation of vision-only localization in strongly periodic layouts.
  If two sites are visually indistinguishable within the available field of view, image information alone may not be sufficient to determine the intended physical site.
