### Checklist

**Essential Checks**
- [ ] Have you merged the most recent `develop`/`master` branch into this one?
- [ ] Is your pull request to merge this branch into the correct `develop`/`master` branch?

Merging in `develop`:

(with working branch active)
```
git checkout develop
git pull
git checkout <your-branch>
git merge develop
```

Merging in `master`:

```
git checkout master
git pull
git checkout <your-branch>
git merge master
```

### Summary of Changes

> Write a summary of the chages in this PR here.

### ODD Updates

> Do these updates need to be accompanied by documentation in the ODD? Do they update any parameters that we use in the model? Review the sections in the ODD in the `docs/odd` folder and write a sentence about any updates here.

### Evaluation of Model & Output

> (Intended for the develop branch) In the absence of tests, write the command you used to run the model. Write the file path location of any output files you manually reviewed to validate proper functionality, and what you were looking for while validating.

### Changes in Related Code

> Write a summary of any code changes indirectly related to this new functionality

### Issues Closed

> Paste URLs to issues closed with this PR, if any.

### Coverage Reports

```
(intended for stable branch)
PASTE COVERAGE REPORT HERE
NUMBA_DISABLE_JIT=1 python -m pytest --pyargs src --cov=src --cov-config=.coveragerc
```

### Other Notes

> Write any other notes or thoughts here

### Summary of Review Process

1. Person `ALPHA`, the owner or person who wrote most of the code, creates a PR and completes the above changes.
2. `ALPHA` assigns person `BETA` as a reviewer.
3. `BETA` reviews and tests code. They add comments on the PR. If they're code specific, add discussions inline with code. If they're abstract add a general comment on the PR.
4. `ALPHA` addresses `BETA`'s comments by:
    1. committing new code to this branch
    2. clarifying the purpose/scope of the PR
    3. creating a new issue that will be addressed in the future
5. `BETA` reviews `ALPHA`'s code and comments to see if they address their issues. `BETA` closes the discussions if `ALPHA`'s changes are adequate.
6. Steps 3-5 repeat until there are no open discussions.
7. `BETA` completes the review and presses the MERGE button and completes the PR.
