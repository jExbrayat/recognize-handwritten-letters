# Good practices

- doctest everything
- reproducible plot functions
  - that return plt.Figure instead of plt.show() (so that it is easier to plot **and** save)
- make several functions for data loading
  - create a lightweight "sample" dataset, to prototype everything and ensures it works
    correctly (because a standard dataset is huge, so it takes time)
- use notebook as less as possible
  - harder to version / collaborate / test
  - ideally, should be used only for documentation examples
  - no training should be launched from it
- formatting & linting
- GitHub CI
- 
