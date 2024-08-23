## .GIT

`.GIT` IS A HIDDEN FOLDER IN `PWD` WHICH TELLS THE GIT TOOL THAT FOLDER WITH `.GIT` IS A FOLDER TO BE MANAGED AS A PART OF VERSION CONTROL SYSTEM
```sh
pwd -   list present working directory
ls  -la -   lists all folders, including hidden
IF .git FOLDER FOUND
    THIS INDICATES THAT THE pwd is a Vesrion Controlled directory by git
```

## Cloning

We can clone any git-hub repo in three ways:

### HTTPS 

```sh
https://github.com/shripadB29/Repository_Edunet.git
```


### SSH
```sh
git@github.com:shripadB29/Repository_Edunet.git
```
NEEDS ``PUBLIC KEY`` IN GIT HUB ACCOUNT


### GIT HUB CLI
```sh
gh repo clone shripadB29/Repository_Edunet
```