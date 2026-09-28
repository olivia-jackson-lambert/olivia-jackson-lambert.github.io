-- 1. Puts the page's cover image (the `image` front-matter field, served as the
--    lighter WebP copy) at the top of the article, under the title block.
-- 2. Renders the `tags` front-matter list as a chip row at the foot of the page.

local function escape(s)
  return s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;")
end

function Pandoc(doc)
  if not FORMAT:match("html") then
    return nil
  end
  local changed = false

  local image = doc.meta.image and pandoc.utils.stringify(doc.meta.image)
  if image and image:match("/thumbs/[^/]+%.png$") then
    local webp = image:gsub("/thumbs/([^/]+)%.png$", "/thumbs/web/%1.webp")
    table.insert(doc.blocks, 1, pandoc.RawBlock("html",
      '<figure class="cs-cover"><img src="' .. webp .. '" alt="" width="900" height="600" fetchpriority="high"></figure>'))
    changed = true
  end

  local tags = doc.meta.tags
  if tags and #tags > 0 then
    local parts = {
      '<div class="page-tags">',
      '<div class="page-tags-label">Also covered</div>',
      '<div class="chips">',
    }
    for _, tag in ipairs(tags) do
      parts[#parts + 1] = '<span class="chip">' .. escape(pandoc.utils.stringify(tag)) .. "</span>"
    end
    parts[#parts + 1] = "</div></div>"
    doc.blocks[#doc.blocks + 1] = pandoc.RawBlock("html", table.concat(parts))
    changed = true
  end

  if changed then return doc end
  return nil
end
