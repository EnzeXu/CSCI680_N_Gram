public class DesignDocument extends CouchDocument { private final JSONObject fulltext; public DesignDocument(final JSONObject json) throws JSONException { super(json); if (!getId().startsWith("_design/")) { throw new IllegalArgumentException(json + " is not a design document"); } fulltext = json.optJSONObject("fulltext"); } public DesignDocument(final CouchDocument doc) throws JSONException { this(doc.json); } public View getView(final String name) throws JSONException { if (fulltext == null) return null; final JSONObject json = fulltext.optJSONObject(name); return json == null ? null : new View(getId() + "/" + name, json); } public Map<String, View> getAllViews() throws JSONException { if (fulltext == null) return Collections.emptyMap(); final Map<String, View> result = new HashMap<>(); final Iterator<?> it = fulltext.keys(); while (it.hasNext()) { final Object key = it.next(); final String name = (String) key; final View view = getView(name); if (view != null) { result.put(name, view); } } return result; } }