-- Renders the `tags` front-matter list as a chip row at the foot of the page.
-- Cards in listings show only the broad `categories`; the granular list lives here.

local function escape(s)
  return s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;")
end

function Pandoc(doc)
  local tags = doc.meta.tags
  if not tags or #tags == 0 then
    return nil
  end

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
  return doc
end
