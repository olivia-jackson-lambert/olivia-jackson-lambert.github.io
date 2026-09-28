-- 1. Renders the `tags` front-matter list as a chip row at the foot of the page.
-- 2. Adds a "Next" link to the end of each case study, in the order given by
--    `case-studies` in _quarto.yml. Pages are matched by their input file name.

local function escape(s)
  return s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"):gsub('"', "&quot;")
end

local function next_block(doc)
  local list = doc.meta["case-studies"]
  if not list then return nil end
  local input = (quarto and quarto.doc and quarto.doc.input_file) or PANDOC_STATE.input_files[1] or ""
  local stem = input:match("([^/]+)%.qmd$")
  if not stem then return nil end
  local n = #list
  for i, item in ipairs(list) do
    local href = pandoc.utils.stringify(item.href)
    if href:match("/" .. stem:gsub("%-", "%%-") .. "%.html$") then
      local nx = list[(i % n) + 1]
      local t = escape(pandoc.utils.stringify(nx.title))
      local h = pandoc.utils.stringify(nx.href)
      local img = pandoc.utils.stringify(nx.image)
      return pandoc.RawBlock("html",
        '<nav class="cs-next" aria-label="Next case study">' ..
        '<a href="' .. h .. '">' ..
        '<span class="cs-next-text"><span class="cs-next-label">Next</span>' ..
        '<span class="cs-next-title">' .. t .. '</span></span>' ..
        '<span class="cs-next-img"><img src="' .. img .. '" alt="" width="900" height="600" loading="lazy" decoding="async"></span>' ..
        '</a></nav>')
    end
  end
  return nil
end

function Pandoc(doc)
  if not FORMAT:match("html") then
    return nil
  end
  local changed = false

  -- page titles read "Title · Olivia Jackson Lambert" unless a page sets its own
  local title = doc.meta.title and pandoc.utils.stringify(doc.meta.title) or ""
  if not doc.meta.pagetitle and title ~= "" then
    doc.meta.pagetitle = pandoc.MetaString(title .. " · Olivia Jackson Lambert")
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

  local nb = next_block(doc)
  if nb then
    doc.blocks[#doc.blocks + 1] = nb
    changed = true
  end

  if changed then return doc end
  return nil
end
