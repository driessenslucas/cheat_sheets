# .NET API Development Cheat Sheet

## Project Setup
```bash
# Create a new Web API project
dotnet new webapi -n ProjectName

# Restore dependencies
dotnet restore

# Run the application
dotnet run

# Build the project
dotnet build

# Add a NuGet package
dotnet add package PackageName

# Run tests
dotnet test
```

---

## Project Structure
```
ProjectName/
├── Controllers/           # API controllers (e.g., WeatherForecastController.cs)
├── Models/                # Data models
├── Services/              # Business logic
├── Repositories/          # Data access logic
├── Program.cs             # Entry point for the app
├── appsettings.json       # App configuration
├── Startup.cs             # Middleware & services (pre .NET 6)
└── Properties/launchSettings.json  # Debug settings
```

---

## Coding Best Practices
### 1. **Project Organization**
- Use **Domain-Driven Design** (Models, Services, Repositories).
- Separate business logic from controllers.
- Keep controllers slim; delegate logic to services.

### 2. **Dependency Injection**
Register services in `Program.cs` (or `Startup.cs` for pre-.NET 6):
```csharp
builder.Services.AddScoped<IMyService, MyService>();
```

### 3. **Configuration**
Use `appsettings.json` for configurations:
```json
{
  "ConnectionStrings": {
    "DefaultConnection": "YourConnectionString"
  }
}
```
Access configuration in code:
```csharp
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection");
```

### 4. **Error Handling**
Use middleware for centralized exception handling:
```csharp
app.UseExceptionHandler("/error");
```
Example global exception handler:
```csharp
app.Use(async (context, next) =>
{
    try
    {
        await next();
    }
    catch (Exception ex)
    {
        context.Response.StatusCode = 500;
        await context.Response.WriteAsync("An error occurred.");
    }
});
```

---

## API Development
### 1. **Controller Basics**
- Annotate controllers with `[ApiController]` and `[Route]`:
```csharp
[ApiController]
[Route("api/[controller]")]
public class MyController : ControllerBase
{
    [HttpGet]
    public IActionResult Get() => Ok("Hello World");
}
```

### 2. **Action Results**
- Common return types:
  ```csharp
  return Ok(data);              // 200 OK
  return NotFound();            // 404 Not Found
  return BadRequest("Error");   // 400 Bad Request
  return StatusCode(500);       // Custom Status Code
  ```

### 3. **Model Validation**
- Use `DataAnnotations` for validation:
  ```csharp
  public class MyModel
  {
      [Required]
      public string Name { get; set; }

      [Range(1, 100)]
      public int Age { get; set; }
  }
  ```
- Validate in controller:
  ```csharp
  [HttpPost]
  public IActionResult Create([FromBody] MyModel model)
  {
      if (!ModelState.IsValid)
          return BadRequest(ModelState);
      return Ok();
  }
  ```

### 4. **Routing**
- Define routes with HTTP verbs:
  ```csharp
  [HttpGet("{id}")]  // GET api/mycontroller/{id}
  public IActionResult Get(int id) => Ok(id);
  ```

---

## Authentication & Authorization
### Add Authentication Middleware
```csharp
builder.Services.AddAuthentication("Bearer")
    .AddJwtBearer(options =>
    {
        options.Authority = "https://your-auth-server";
        options.Audience = "your-api";
    });

app.UseAuthentication();
app.UseAuthorization();
```

### Protect Endpoints
```csharp
[Authorize]
[HttpGet]
public IActionResult SecureEndpoint() => Ok("Secure Data");
```

---

## Entity Framework Core (EF Core)
### Add EF Core
```bash
dotnet add package Microsoft.EntityFrameworkCore
dotnet add package Microsoft.EntityFrameworkCore.SqlServer
```

### Configure Database Context
```csharp
public class AppDbContext : DbContext
{
    public DbSet<MyModel> Models { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
        => options.UseSqlServer("YourConnectionString");
}
```

### Migrations
```bash
# Add migration
dotnet ef migrations add InitialCreate

# Update database
dotnet ef database update
```

---

## Versioning
### Add Versioning
```csharp
builder.Services.AddApiVersioning(options =>
{
    options.ReportApiVersions = true;
    options.AssumeDefaultVersionWhenUnspecified = true;
    options.DefaultApiVersion = new ApiVersion(1, 0);
});
```

### Use Versioned Routes
```csharp
[ApiVersion("1.0")]
[Route("api/v{version:apiVersion}/[controller]")]
public class MyController : ControllerBase { }
```

---

## Logging
### Add Logging
```csharp
builder.Services.AddLogging();

ILogger<MyController> logger = app.Services.GetRequiredService<ILogger<MyController>>();
logger.LogInformation("Application started.");
```

---

## Useful Commands
```bash
# List installed .NET SDKs
dotnet --list-sdks

# Run the app with specific environment
dotnet run --launch-profile "Development"
```

---

## Testing
### Add Test Project
```bash
dotnet new xunit -n ProjectName.Tests
```

### Write Tests
Example controller test with `Moq`:
```csharp
[Fact]
public async Task Get_ReturnsOk()
{
    var controller = new MyController();
    var result = controller.Get();
    Assert.IsType<OkObjectResult>(result);
}
```

---

## Helpful Resources
- [.NET Documentation](https://docs.microsoft.com/dotnet/)
- [ASP.NET Core Documentation](https://docs.microsoft.com/aspnet/core/)

