# Contributing
We love contributions! We've compiled these docs to help you understand our contribution guidelines. If you still have questions, please [contact us](mailto:datascience@nhs.net), we'd be super happy to help.

## Contents of this file

- [Code of conduct](#code-of-conduct)
- [Folder structure](#folder-structure)
- [Commit hygiene](#commit-hygiene)
- [Updating Changelog](#updating-changelog)

## Code of Conduct
Please read [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) before contributing.

## Commit hygiene

Please see the [GDS guidelines](https://gds-way.digital.cabinet-office.gov.uk/standards/source-code/working-with-git.html#working-with-git), which describe general recommendations regarding commits and commit messages.

## Updating the Changelog

If you open a GitHub pull request on this repo, please update `CHANGELOG.md` to reflect your contribution.

Add your entry under `Unreleased` as:
- `Breaking changes`
- `Enhancements`
- `Bug fixes`
- `Documentation`
- `Miscellaneous`

Internal changes to the project that are not part of the public API do not need changelog entries, for example fixing the CI build server.

These sections follow [semantic versioning](https://semver.org/spec/v2.0.0.html), where:

- `Breaking changes` corresponds to a `major` (1.X.X) change.
- `Enhancements` corresponds to a `minor` (X.1.X) change.
- `Bug fixes` corresponds to a `patch` (X.X.1) change.

See the [`CHANGELOG.md`](./CHANGELOG.md) for an example for how this looks. However, note that the project is currently using major version zero, which means that the public API is not yet fully stable, and that the standard semantic versioning conventions do not apply.
