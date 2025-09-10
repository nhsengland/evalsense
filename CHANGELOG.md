# Changelog

All notable changes to this project will be documented in this file.

Instructions on how to update this Changelog are available in the `Updating the Changelog` section of the [`CONTRIBUTING.md`](./CONTRIBUTING.md). This project follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Breaking changes

- The API and the internal logic of dataset managers have been updated. The `priority` argument to the constructor is no longer used. Instead, all classes inheriting from `DatasetManager` should define a class-level `priority` attribute. Additionally, the file-based dataset managers, such as the built-in `AciBenchDatasetManager`, now preprocess all splits when retrieving the dataset. This is more expensive, but prevents issues with loading different splits.

### Enhancements

- Implemented support for meta-evaluation of different evaluation methods using `MetaResultAnalyser`.
- It is now possible to construct dataset managers using the `DatasetManager.create(...)` class method, which automatically instantiates the dataset manager associated with the passed dataset name. This is facilitated by an internal registry of dataset managers.
- A new `HuggingFaceDatasetManager` enables loading arbitrary datasets from Hugging Face Hub.

### Bug fixes

- Fix method_filter_fun behaviour in MetricCorrelationAnalyser.

### Documentation

- None

### Miscellaneous

- EvalSense now places the xet cache under EvalSense cache directory instead of the default HuggingFace directory.

## v0.1.3

### Breaking changes

- Make local model dependencies optional during package installation. If you are using EvalSense with local models, you should install `evalsense[all]` or `evalsense[local]` instead of `evalsense`.

## v0.1.2

Technical release: no user-facing changes.

## v0.1.1

Technical release: no user-facing changes.

## v0.1.0

Initial release.
