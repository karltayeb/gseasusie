jax_env <- basilisk::BasiliskEnvironment(envname="jax_env",
    pkgname="gseasusie",
    packages=c("numpy==1.19.2", "pandas==1.2.1", "scipy==1.6.0"),
    pip=c("jax==0.3.10", "jaxlib==0.3.10")
)