# 시각화 열기와 GitHub Pages 게시

## 지금 파일로 확인

저장소를 clone한 뒤 `docs/index.html` 또는 `docs/viewer/index.html`을 브라우저에서 열 수 있습니다. Viewer는 약11MB self-contained HTML이며 data/font를 포함하므로 CDN이나 모델/GPU가 필요 없습니다. GitHub의 HTML 파일 화면에서는 코드로 표시되므로 파일을 다운로드하거나 clone해서 여세요.

## GitHub Pages

Repository Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/docs` → Save.

게시가 완료되면 예상 주소는 `https://decembix.github.io/MoE-Compression-analysis/`이며 Viewer는 `https://decembix.github.io/MoE-Compression-analysis/viewer/`입니다. 이 문서는 Pages 설정이나 배포 완료를 보장하지 않으며 GitHub의 Pages 상태 화면에서 실제 게시를 확인하세요. `docs/.nojekyll`이 정적 HTML 게시를 위한 파일입니다. Site의 analysis 링크는 GitHub의 rendered Markdown으로 연결됩니다.

공식 안내: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## 로컬 HTTP preview (선택)

```bash
python -m http.server 8000 --directory docs
```

브라우저에서 `http://localhost:8000`을 엽니다. 처음에는 기존 self-contained viewer를 그대로 제공하며, 향후 layer별 JSON lazy loading으로 초기 로딩을 줄일 수 있습니다.
