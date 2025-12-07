# Contributing to kerasnip

This outlines how to propose a change to kerasnip. For more detailed
info about contributing to this, and tidyverse packages, please see the
[**development contributing guide**](https://rstd.io/tidy-contrib).

### Fixing typos

Small typos or grammatical errors in documentation may be edited
directly using the GitHub web interface, so long as the changes are made
in the *source* file.

- YES: you edit a roxygen comment in a `.R` file below `R/`.
- NO: you edit an `.Rd` file below `man/`.

### Prerequisites

Before you make a substantial pull request, you should always file an
issue and make sure someone from the team agrees that it’s a problem. If
you’ve found a bug, create an associated issue and illustrate the bug
with a minimal [reprex](https://www.tidyverse.org/help/#reprex).

### Pull request process

- We recommend that you create a Git branch for each pull request
  (PR).  
- Look at the **GitHub Actions** build status before and after making
  changes. The `README` contains badges for the continuous integration
  services used by the package.  
- New code should follow the tidyverse [style
  guide](http://style.tidyverse.org). You can use the air package to
  apply these styles. You can format your code automatically by
  commenting `/style` on your PR.  
- We use [roxygen2](https://cran.r-project.org/package=roxygen2), with
  [Markdown
  syntax](https://cran.r-project.org/web/packages/roxygen2/vignettes/rd-formatting.html),
  for documentation.  
- We use [testthat](https://cran.r-project.org/package=testthat).
  Contributions with test cases included are easier to accept.  
- For user-facing changes, add a bullet to the top of `NEWS.md` below
  the current development version header describing the changes made
  followed by your GitHub username, and links to relevant
  issue(s)/PR(s).

### Code of Conduct

Please note that this project is released with a [Contributor Code of
Conduct](https://davidrsch.github.io/kerasnip/dev/CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

### See tidyverse [development contributing guide](https://rstd.io/tidy-contrib) for further details.
