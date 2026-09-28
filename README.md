# CollisionSplatting project page

Static site (plain HTML/CSS/JS, no build step): `index.html`, `static/` (css, js, images, videos).
KaTeX and Google Fonts load from CDNs.

## Before publishing
1. Add the camera-ready PDF as `paper.pdf` next to `index.html` (the "Read the paper" button links to it),
   or change that link to the arXiv / IEEE Xplore page.
2. Fill in the author links (`href="#"` in the author list of `index.html`).
3. Make sure https://github.com/machines-in-motion/CollisionSplatting is public (the "Get the code" button and the poster QR point there).

## Preview locally
    python -m http.server 8000     # then open http://localhost:8000

## Put it online (GitHub Pages)
**Option A — gh-pages branch of the code repo** (URL: https://machines-in-motion.github.io/CollisionSplatting/)

    cd paper-website
    git init -b gh-pages && git add . && git commit -m "Project page"
    git remote add origin git@github.com:machines-in-motion/CollisionSplatting.git
    git push -u origin gh-pages

Then on GitHub: Settings → Pages → Source: "Deploy from a branch" → Branch `gh-pages`, folder `/ (root)` → Save.
The site is live after ~1 minute. Updates: commit and push to `gh-pages` again.

**Option B — separate repo** `CollisionSplatting-website` (URL: https://machines-in-motion.github.io/CollisionSplatting-website/):
same steps with branch `main`, and pick `main` / root in Settings → Pages.

Custom domain (optional): add a `CNAME` file containing the domain, and a DNS CNAME record pointing to
`machines-in-motion.github.io`.

Note: Pages on a private repo requires a paid GitHub plan; a public repo works on any plan.
