# Open Code Checklist

To ensure that the code we publish is appropraite, clear and reusable we require a set way of working when pushing code to private and public repoistories. It is difficult to set any precise but generic standards and so we ask that the following checklist is considered with mandatroy items filled out.

## When publishing your code you need to make sure:

### You’re clear about who owns the code and how others can use it

- [x] Does your code have an appropriate licence and copyright notice? (**Mandatory**)
- [x] Is there a README and does it document intended purpose? (**Mandatory**)
- [x] Is the README clear and concise? (**Optional** - use analytics unit template or [example](https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md))
- [x] Do you need to consider MHRA 'software as a medical device' guidance? (**Mandatory** - use [flowchart](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/999908/Software_flow_chart_Ed_1-08b-IVD.pdf))
  - No
- [x] Who has responsibility for ongoing support and communications for the code? (**Optional** Best practice to assign, else state the code is not maintained and when the last update was)
  - [datascience@nhs.net](mailto:datascience@nhs.net) is responsible for the ongoing support and communication about the code
- [x] Has a responsible disclosure process for security issues been defined? (**Mandatory** - could use github issues)
  - See [SECURITY.md](https://github.com/nhsengland/evalsense/blob/main/SECURITY.md).
- [x] Who will address identified issues and security concerns? (**Mandatory**. - assign monitoring of above disclosure process)
  - @[adamdejl](https://github.com/adamdejl) and [datascience@nhs.net](mailto:datascience@nhs.net) will monitor and address any reported security issues.
- [ ] Has semantic versioning been used? (**Optional**)
  - Not currently applicable: this project is a work in progress and doesn't have any official releases yet.
- [x] Have [contribution guidelines](https://github.com/alphagov/govuk-frontend/blob/master/CONTRIBUTING.md) or [PR style guides](https://github.com/alphagov/styleguides/blob/master/pull-requests.md) been included? (**Optional**)
- [x] Are package dependencies and libaries documented with versions? (**Optional**)
- [x] Has the code been linked to any published outputs so others can easily link to the code? (**Optional**)
  - Not applicable

### You do not release information that should remain closed

- [x] Does the code include any sensitive, personal, secret or top secret data/information? (**Mandatory**)
  - No
- [x] Does the code include any unreleased policy? (**Mandatory**)
  - No
- [x] Does the code include business sensitive algorithms (e.g. finance allocations)? (**Mandatory**)
  - No
- [x] Has written permission been obtained for any data stored from the data owner? (**Mandatory**)
  - No third-party data is being stored in this repository.
- [x] Are owners of services which the code fulfils aware of the release? (**Mandatory**)
- [x] Are any data transfers conducted safely and securely? (**Mandatory**)
- [x] Are any credentials contained in the source code? (**Mandatory** - check in both current version and git history)
  - No
- [x] Are any secret keys contained in the source code? (**Mandatory** - check in both current version and git history)
  - No
- [x] Are any SQL server addresses or connection strings in the source code? (**Mandatory** - check in both current version and git history)
  - The code is not interfacing with any external databases.
- [x] Are the commit messages informative? (**Optional**)
- [x] Do the commit messages include any sensitive information (e.g. names)? (**Mandatory**)
  - No, only names of the commit authors
- [x] Does the git history contain any sensitive information (e.g. at one time real data or credentials were in the code but have since been removed) (**Mandatory**)
  - No
- [x] Have notebook outputs been removed/checked for sensitive information? (**Mandatory** - check but some appropriate outputs maybe useful: [Example](https://github.com/best-practice-and-impact/govcookiecutter/blob/main/%7B%7B%20cookiecutter.repo_name%20%7D%7D/.pre-commit-config.yaml))
  - Pre-commit hooks are set to remove any notebook outputs before committing.
- [x] Is configuration written as code and separated from analytical code? (**Optional**)
  - Configuration for datasets is stored in separate `.yml` files.
- [x] have you checked any screenshots or figures in your outputs and documentation for information that shouldn't be released? (**Mandatory**)
  - No screenshots or figures are stored in this repository.

### You store it in a repository managed by your department (to make licensing/copyright clear)

- [x] Is the code version controlled using GIT or similar? (**Optional**)
- [x] Is the code stored in your organisational GitHub account? Is it the same organisation that funds the relevant staff time? (**Optional**)

### Any third-party tools you use to host or manage your code follow the National Cyber Security Centre’s cloud security guidance

- [x] Are third party tools used within the code? (**Mandatory** check. Best practice is to keep an inventory)
  - Yes, see [pyproject.toml](https://github.com/nhsengland/evalsense/blob/main/pyproject.toml) and [package.json](https://github.com/nhsengland/evalsense/blob/main/guide/package.json).
- [x] If so do they adhere to the NCSC's [Cloud Security Principles](https://www.ncsc.gov.uk/collection/cloud-security/implementing-the-cloud-security-principles)? (**Mandatory**)

### An internal code review has been completed

- [x] Has a colleague reviewed the code for sensitive data content and security vulnerabilities? (**Mandatory** - Best practice is to record automated code quality and security tools used)
  - Yes, we audited the environment using `pip-audit`, performed a manual search for CVEs and vetted the authors and the manntenance status of the used packages.
- [x] Has a code quality review been completed focussing on the end usability and clarity? (**Optional** - consider runing through the [example](https://best-practice-and-impact.github.io/qa-of-code-guidance/checklist_higher.html) or similar code quality checklist)
- [ ] Has the code been assessed for its [levels of RAP](https://github.com/NHSDigital/rap-community-of-practice/blob/main/docs/introduction_to_RAP/levels_of_RAP.md) (Reproducible Analytical Pipeline)? (**Optional**)
- [ ] Has the code undergone some level of testing. The level of testing required will depend on the specific code and use-case but as minimum it should work in a fresh environment with artificial data. (**Optional**)
