# Create the GitHub repository

Use **`NeurALPS-v2`** as the repository name. Suggested description:

> Domain- and connection-centric representation learning for nonribosomal peptide synthetase assembly lines.

Suggested topics: `nrps`, `protein-language-models`, `protein-engineering`, `biosynthesis`, `self-supervised-learning`, `pytorch`, `representation-learning`.

## First local commit

Extract the archive and open a terminal inside its `NeurALPS-v2` directory. Initialize Git only if this is not already a clone:

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Initialize NeurALPS v2 research reference"
```

Use your normal Git author configuration. The scaffold does not set an author name or email.

## Create and push with GitHub CLI

After authenticating with `gh auth login`, this command creates a private repository in the authenticated account and pushes the current commit:

```bash
gh repo create NeurALPS-v2 --private --source=. --remote=origin --push
```

For an organization, use `ORGANIZATION/NeurALPS-v2` in place of the repository name. Use `--public` if you intend to publish it publicly. The command's behavior is documented in the [GitHub CLI manual](https://cli.github.com/manual/gh_repo_create).

## Create and push through the website

Create an empty repository on GitHub with the desired visibility. Leave GitHub's README, license and gitignore initialization options unchecked because this folder already contains its working files. Copy the remote URL GitHub supplies, then use it in place of the quoted placeholder:

```bash
git remote add origin "PASTE_THE_REPOSITORY_URL_HERE"
git push -u origin main
```

These are setup instructions. This scaffold has no remote repository attached and has not been pushed.

## Repository settings

Set the description and topics, enable Issues, and allow the `Reference checks` workflow to run. After the first successful hosted run, choose its checks for the `main` branch's merge rules if appropriate for your collaboration model.

Before a tagged public release, fill in the actual repository URL and maintainer-approved author metadata in `CITATION.cff` and select the project license. Add a paper DOI only when there is a real manuscript record. Keep model/data availability statements tied to actual release artifacts.

The workflow follows the [official GitHub Python CI guidance](https://docs.github.com/en/actions/tutorials/build-and-test-code/python) and uses [setup-python](https://github.com/actions/setup-python). The Python package uses a [`pyproject.toml` configuration](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
