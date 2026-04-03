if __name__ == "__main__":
    import runpy

    runpy.run_module("services.ml.pitch_extraction", run_name="__main__")
else:
    from importlib import import_module

    def _load_impl():
        return import_module("services.ml.pitch_extraction")

    def __getattr__(name):
        return getattr(_load_impl(), name)

    def __dir__():
        return sorted(set(globals()) | set(dir(_load_impl())))
