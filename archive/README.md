# 归档：历史路由缓存

这些是 `route.py` 在不同版本下的缓存文件，格式相同（问题 → 模型选中的章节），
只是问的时候目录不一样。**运行时一个都不读**，留着是因为它们是 ⑥⑦ 两节结论的证据。

| 文件 | 是什么 | 还用不用 |
|---|---|---|
| `route_cache_e8e70fe3.json` | 目录里**开了反查词**（`HINT`）的版本 | 复现「反查词有没有用」时切过去 |
| `route_cache_8db57d68.json` | 删掉 `No.` 编号之前的那版，模型会揉号（`16.1` 想选 `12.1`） | 只作历史 |
| `route_cache_oral.json` | 极短口语 13 题的缓存 | 只作历史 |
| `route_cache_hint.json` | 反查词第一版的缓存，被 `e8e70fe3` 取代 | 只作历史 |
| `route_cache.json` | 最早的固定文件名，后来改成按目录 md5 命名 | `exp/alias.py`、`exp/hint.py` 拿它当「旧基线」 |
| `route_cache_old.json` | `refresh_route.py` 自动备份出来的上一版 | 只作历史 |
| `routes.json` | 更早的缓存格式，被 `route_cache*` 取代 | 只作历史 |

**当前在用的只有根目录那两个**（不在本目录）：

```
route_cache_300d7be7.json     ROUTE_MODE=sec 时读它；文件名是目录 MENU 的 md5
allsub_cache_ec678ff4.json    小节直选（默认模式）的缓存
```

文件名带哈希是有意的：目录改一个字，哈希就变，缓存自然失效。以前用固定文件名时，
改完目录还在吃旧缓存，看起来像「改了没效果」——这个坑记在 ⑥ 节。
