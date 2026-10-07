# Changelog

## [1.1.0](https://github.com/IIAS-Research/iaml/compare/v1.0.2...v1.1.0) (2026-10-07)


### Features

* compile configured metric objectives ([e1e3c4c](https://github.com/IIAS-Research/iaml/commit/e1e3c4c73fe5545166a1f963dca1e13d48a963e0))
* compile recipes into execution trees ([d6a9af7](https://github.com/IIAS-Research/iaml/commit/d6a9af7f99dc3750530821eae24f4d3e1f51ad9b))
* compose independent pipeline recipes ([d43a10b](https://github.com/IIAS-Research/iaml/commit/d43a10b3f7857c41d13f3c7a25f601471a168425))
* describe and compare recipes ([c29f854](https://github.com/IIAS-Research/iaml/commit/c29f8544aa8d1ba4aab4a7c18317349e8d1ee946))
* describe datasets without training ([a4f2602](https://github.com/IIAS-Research/iaml/commit/a4f2602bc8519702966d25f366b2be1f7ed5aed7))
* distinguish configured Bayesian search spaces ([784f0af](https://github.com/IIAS-Research/iaml/commit/784f0afba8b5ff928191ec20b066abc322cb4024))
* edit recipes and component families ([0443285](https://github.com/IIAS-Research/iaml/commit/044328585daf6077bb2d32cdd2deb26e2ba74624))
* execute configured explanations on demand ([4ef2684](https://github.com/IIAS-Research/iaml/commit/4ef26840366da27a5ffd64e0fec2299a363b7739))
* expose the default pipeline as a recipe ([885c7ec](https://github.com/IIAS-Research/iaml/commit/885c7ec25d3abd361c1c2519b36671d701b75a45))
* honor explicit random seed domains ([d2f03e4](https://github.com/IIAS-Research/iaml/commit/d2f03e43b9e9960d4d72412f27e3b7c8ed87382a))
* introduce fixed values and numeric domains ([73e994e](https://github.com/IIAS-Research/iaml/commit/73e994eb7d56441333a4c233bda879d6cce1fbd1))
* optimize declared training parameters ([3eb4628](https://github.com/IIAS-Research/iaml/commit/3eb462822f1216325e4e6986ea03464762340be5))
* preserve metric aliases and report fold coverage ([fb84207](https://github.com/IIAS-Research/iaml/commit/fb842079cddfcc3e47d18d05e4b750705a9eb1cd))
* reconstruct recipes as Python code ([4a2ec1d](https://github.com/IIAS-Research/iaml/commit/4a2ec1dbe9c0959c7bd61f551b3474db98946be0))
* report resolved study snapshots ([f82c744](https://github.com/IIAS-Research/iaml/commit/f82c7449af0dd1b71d94142a634885ee8367715c))
* restrict genetic swaps to declared alternatives ([5e04552](https://github.com/IIAS-Research/iaml/commit/5e045527c87d4c9e8984d637b071a152b7cbb787))
* track parameter policies in caches ([cf7157b](https://github.com/IIAS-Research/iaml/commit/cf7157ba3253f8757d436867b307f113abba0327))
* train from attached pipeline recipes ([141dda5](https://github.com/IIAS-Research/iaml/commit/141dda5073e1edfb843e34aac50996302c7bbd93))


### Bug Fixes

* align binary curves with fitted classes ([d1156f0](https://github.com/IIAS-Research/iaml/commit/d1156f019b780990b921c8620a2a10d017deba78))
* align permutation scoring with the main metric ([1d94a57](https://github.com/IIAS-Research/iaml/commit/1d94a5736c1b0524e4de648d7961b5f65c40af49))
* compare final candidates on consistent evaluations ([725422a](https://github.com/IIAS-Research/iaml/commit/725422ae4e52752ecdf27438c383e463e5dad979))
* encode categorical features in minimal pipelines ([c019e2e](https://github.com/IIAS-Research/iaml/commit/c019e2e7512c7fd059e305b2ad763866b965103f))
* handle open numeric bounds ([97d09b4](https://github.com/IIAS-Research/iaml/commit/97d09b4ef2624783a795c0b944d8e10700fe0f65))
* honor configured MICE parameters ([d6974a8](https://github.com/IIAS-Research/iaml/commit/d6974a851972f5f4f869997d8ce768ff1aeaaace))
* honor study configuration in baseline ([139b24d](https://github.com/IIAS-Research/iaml/commit/139b24d6106d4ba2a9d7ca716a09eec94efc92e9))
* improve CatBoost hyperparameter search spaces ([29f33b4](https://github.com/IIAS-Research/iaml/commit/29f33b4d340d6bba0b5553a7539a307f7ef7f015))
* keep Sphinx caches outside published documentation ([c81c1c3](https://github.com/IIAS-Research/iaml/commit/c81c1c3d1384a9c3742190d144abfbb64b55e558))
* make default feature selection optional ([23f3583](https://github.com/IIAS-Research/iaml/commit/23f35834d40db2a2281d380149a32e5873789164))
* preserve evaluations across sliding search stages ([e4e4433](https://github.com/IIAS-Research/iaml/commit/e4e44335fb29e4149b096cd4fd59deada09822a4))
* preserve inputs during inference ([ad5062d](https://github.com/IIAS-Research/iaml/commit/ad5062d99f65446af8ba0757e08d40023c817f27))
* reduce survival forest memory usage ([53c93f2](https://github.com/IIAS-Research/iaml/commit/53c93f236ae3446d637b12baa2ed0c074e08284b))
* render box plots with the axes API ([57d2e8b](https://github.com/IIAS-Research/iaml/commit/57d2e8b879f2470b8383cb23625d51587b7e8543))
* respect automatic subsampling settings ([1e9cc4e](https://github.com/IIAS-Research/iaml/commit/1e9cc4ee8c3dd36a705cb7cf7266aa3b12791aee))
* support decision scores for binary AUROC ([244304f](https://github.com/IIAS-Research/iaml/commit/244304f0ed677f87b48f7b7ab9a4417a20440970))
* use supported MICE imputation fallback ([757e296](https://github.com/IIAS-Research/iaml/commit/757e296ae97b8b7a10c6614e878526d5a9217961))


### Performance Improvements

* skip unnecessary SVC probability calibration ([321b962](https://github.com/IIAS-Research/iaml/commit/321b9624842341298abc5b5ce9cde6e1ff789e08))
* start search with a lightweight baseline ([ae75e94](https://github.com/IIAS-Research/iaml/commit/ae75e940425bf94e8b6b25d38d365e3c8855e93e))
* use three folds for internal cross-validation ([f534b19](https://github.com/IIAS-Research/iaml/commit/f534b19187066f8d8d0cfec722ae6a8f1eea6dbd))


### Documentation

* add basic recipe examples ([227b4ca](https://github.com/IIAS-Research/iaml/commit/227b4ca23171c659ff7b2aa47fed2c9690bba957))
* add reusable strategies and study workflows ([ed303a6](https://github.com/IIAS-Research/iaml/commit/ed303a6b9b6377999164529e3d5b9f80a2deac00))
* clarify stage and global search budgets ([3cfde1e](https://github.com/IIAS-Research/iaml/commit/3cfde1e4e03031138d05802354d2b242d4231b8c))
* connect guides and preserve previous URLs ([c0cc94b](https://github.com/IIAS-Research/iaml/commit/c0cc94b0f1411b277c72d56386034e3cbfde1a21))
* document recipe construction and editing ([02fd394](https://github.com/IIAS-Research/iaml/commit/02fd3942cb972cad3b635c4958c76df32c81cf69))
* document study analyses ([7d558c5](https://github.com/IIAS-Research/iaml/commit/7d558c543b373341da2f0b07786b3e34bbf92152))
* introduce interactive pipeline discovery ([5e38f2f](https://github.com/IIAS-Research/iaml/commit/5e38f2febfee7e4df3ad76dce236dd8417318bb1))
* publish the Python example catalogue ([c1ce297](https://github.com/IIAS-Research/iaml/commit/c1ce297d69b40edff88807e9d3f4fca2912c8241))
* update the README and existing user guides ([bdc9521](https://github.com/IIAS-Research/iaml/commit/bdc9521c947eed7b1999b64dad7e47ed6320a255))

## [1.0.2](https://github.com/IIAS-Research/iaml/compare/v1.0.1...v1.0.2) (2026-09-20)


### Documentation

* add Zenodo citation metadata ([9ba4291](https://github.com/IIAS-Research/iaml/commit/9ba42916b5de0b805c006a824c740edf18ebb7dc))

## [1.0.1](https://github.com/IIAS-Research/iaml/compare/v1.0.0...v1.0.1) (2026-09-20)


### Documentation

* add an interactive homepage ([917d481](https://github.com/IIAS-Research/iaml/commit/917d481bf57e3d11a0cff7e60e5dd890971e7212))
* clarify architecture and extensions ([d303163](https://github.com/IIAS-Research/iaml/commit/d3031636a1a0b65c5d74406427e905ebf93f7c41))
* clean up API descriptions ([f3365a9](https://github.com/IIAS-Research/iaml/commit/f3365a970d0bfa372201e68973515220a0591a9a))
* improve user guides and examples ([2306bc2](https://github.com/IIAS-Research/iaml/commit/2306bc2aac21daecc7d136114fbdc1ca7dd61ff3))
* refresh documentation design ([4c826aa](https://github.com/IIAS-Research/iaml/commit/4c826aa296e1e5b692a070303421039f52118b4b))

## [1.0.0](https://github.com/IIAS-Research/iaml/compare/v0.1.1...v1.0.0) (2026-09-19)


### Features

* ci for documentation and versioning ([c1be8ec](https://github.com/IIAS-Research/iaml/commit/c1be8ecbe7f1d5f7cd85a7547763b5ce45177740))


### Bug Fixes

* support Python 3.12 installation ([447a31b](https://github.com/IIAS-Research/iaml/commit/447a31b9879109dd1b65b4e004fbe0ffee347f37))
