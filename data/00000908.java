public class CouchDocument { protected final JSONObject json; private static final String ID = "_id"; private static final String DELETED = "_deleted"; public static CouchDocument deletedDocument(final String id) throws JSONException { final JSONObject json = new JSONObject(); json.put(ID, id); json.put(DELETED, true); return new CouchDocument(json); } public CouchDocument(final JSONObject json) { if (!json.has(ID)) { throw new IllegalArgumentException(json + " is not a document"); } this.json = json; } public String getId() throws JSONException { return json.getString(ID); } public boolean isDeleted() { return json.optBoolean(DELETED, false); } public JSONObject asJson() { return json; } }