// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "ArrayDebug.hpp"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Theseus::Debug
{
  using json = nlohmann::json;
  namespace fs = std::filesystem;

  namespace
  {
    std::string Padded(std::int64_t value)
    {
      std::ostringstream out;
      out << std::setfill('0') << std::setw(6) << value;
      return out.str();
    }

    bool Matches(const std::string &pattern, const std::string &name)
    {
      if (!pattern.empty() && pattern.back() == '*')
      {
        return name.compare(0, pattern.size() - 1, pattern, 0, pattern.size() - 1) == 0;
      }
      return pattern == name;
    }

    void Require(bool condition, const std::string &message)
    {
      if (!condition)
      {
        throw std::runtime_error("Array inspection: " + message);
      }
    }
  }

  struct ArrayInspector::Impl
  {
    json config;
    json context;
    fs::path output;
    std::ofstream manifest;
    std::set<std::string> recorded;
    std::int64_t call = -1;
    std::uint64_t bytes_written = 0;
    std::uint64_t max_bytes = 1024ULL * 1024 * 1024;
    int rank = 0;
    bool enabled = false;
    bool selected = false;
    bool snapshots = false;

    void Append(const json &record)
    {
      manifest << record.dump() << '\n';
      manifest.flush();
      Require(manifest.good(), "cannot write manifest");
    }
  };

  ArrayInspector::ArrayInspector() : impl(new Impl)
  {
    const char *path = std::getenv("THESEUS_ARRAY_DEBUG");
    if (!path || !*path)
    {
      return;
    }
    std::ifstream input(path);
    Require(input.good(), "cannot open configuration " + std::string(path));
    input >> impl->config;
    const auto &config = impl->config;
    Require(config.is_object(), "configuration must be an object");
    const std::set<std::string> keys = {
      "output", "calls", "ranks", "arrays", "snapshots", "max_bytes", "tag"};
    for (auto it = config.begin(); it != config.end(); ++it)
    {
      Require(keys.count(it.key()) != 0, "unknown configuration key " + it.key());
    }
    impl->output = config.at("output").get<std::string>();
    Require(impl->output.is_absolute(), "output must be an absolute directory");
    for (const auto *key : {"calls", "ranks"})
    {
      Require(config.contains(key) && config.at(key).is_array() && !config.at(key).empty(),
              std::string(key) + " must be a nonempty array of nonnegative integers");
      std::set<std::int64_t> seen;
      for (const auto &value : config.at(key))
      {
        Require(value.is_number_integer() && value.get<std::int64_t>() >= 0,
                std::string("invalid ") + key + " selection");
        Require(seen.insert(value.get<std::int64_t>()).second, "duplicate selection");
      }
    }
    Require(config.contains("arrays") && config.at("arrays").is_array() &&
            !config.at("arrays").empty(), "arrays must be a nonempty array of names/prefixes");
    for (const auto &value : config.at("arrays"))
    {
      const auto pattern = value.get<std::string>();
      Require(!pattern.empty() && pattern.find('*') >= pattern.size() - 1,
              "only a trailing wildcard is supported");
    }
    impl->snapshots = config.value("snapshots", false);
    if (config.contains("max_bytes"))
    {
      Require(config.at("max_bytes").is_number_integer() &&
              config.at("max_bytes").get<std::int64_t>() >= 0, "invalid max_bytes");
      impl->max_bytes = config.at("max_bytes").get<std::uint64_t>();
    }
    impl->enabled = true;
  }

  ArrayInspector::~ArrayInspector() = default;

  void ArrayInspector::BeginCall(MPI_Comm comm, double time, int dimension,
                                  int equations, int order, const std::string &gas,
                                  const std::string &flux)
  {
    if (!impl->enabled)
    {
      return;
    }
    Require(!impl->selected, "previous selected call has no EndCall");
    ++impl->call;
    MPI_Comm_rank(comm, &impl->rank);
    int ranks = 0;
    MPI_Comm_size(comm, &ranks);
    const auto &calls = impl->config.at("calls");
    const auto &selected_ranks = impl->config.at("ranks");
    for (const auto &selected_rank : selected_ranks)
    {
      Require(selected_rank.get<int>() < ranks, "selected rank exceeds communicator size");
    }
    impl->selected = std::find(calls.begin(), calls.end(), impl->call) != calls.end() &&
      std::find(selected_ranks.begin(), selected_ranks.end(), impl->rank) != selected_ranks.end();
    if (!impl->selected)
    {
      return;
    }
    if (!impl->manifest.is_open())
    {
      fs::create_directories(impl->output);
      const auto path = impl->output / ("rank" + Padded(impl->rank) + ".jsonl");
      Require(!fs::exists(path), "manifest already exists; use a fresh output directory: " + path.string());
      impl->manifest.open(path);
      Require(impl->manifest.good(), "cannot create manifest " + path.string());
    }
    const std::string device = mfem::Device::Allows(mfem::Backend::CUDA_MASK) ? "cuda"
      : (mfem::Device::Allows(mfem::Backend::HIP_MASK) ? "hip" : "host");
    impl->context = {{"schema", 1}, {"call", impl->call}, {"rank", impl->rank},
      {"ranks", ranks}, {"time", time}, {"dimension", dimension}, {"equations", equations},
      {"order", order}, {"gas", gas}, {"flux", flux}, {"device", device},
      {"tag", impl->config.value("tag", std::string())}};
    impl->recorded.clear();
    auto record = impl->context;
    record["kind"] = "begin";
    record["configuration"] = impl->config;
    impl->Append(record);
  }

  void ArrayInspector::Record(const std::string &name, const mfem::Vector &values,
                              std::initializer_list<const char *> axes,
                              std::initializer_list<std::int64_t> shape)
  {
    if (!impl->selected)
    {
      return;
    }
    bool selected = false;
    for (const auto &pattern : impl->config.at("arrays"))
    {
      selected = selected || Matches(pattern.get<std::string>(), name);
    }
    if (!selected)
    {
      return;
    }
    Require(!name.empty() && name.find_first_not_of(
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") == std::string::npos,
      "unsafe array name");
    Require(axes.size() == shape.size() && shape.size() > 0, "invalid layout for " + name);
    std::int64_t count = 1;
    for (const auto extent : shape)
    {
      Require(extent >= 0 && (extent == 0 || count <= std::numeric_limits<std::int64_t>::max() / extent),
              "invalid shape for " + name);
      count *= extent;
    }
    Require(count == values.Size(), "shape/size mismatch for " + name);
    Require(impl->recorded.insert(name).second, "duplicate array in call: " + name);

    // Copy into private storage without validating the vector's own host copy.
    // Kernels may keep a device write pointer across this observation point;
    // HostRead() would leave a host-valid flag that those later writes cannot clear.
    std::vector<mfem::real_t> host(values.Size());
    values.GetMemory().CopyToHost(host.data(), values.Size());
    const auto *data = host.data();
    json record = impl->context;
    record["kind"] = "array";
    record["name"] = name;
    record["sequence"] = impl->recorded.size() - 1;
    record["axes"] = axes;
    record["shape"] = shape;
    record["size"] = count;
    const std::uint16_t endian_probe = 1;
    const bool little = *reinterpret_cast<const unsigned char *>(&endian_probe) == 1;
    record["dtype"] = std::string(little ? "<f" : ">f") + std::to_string(sizeof(mfem::real_t));
    std::int64_t nan = 0, posinf = 0, neginf = 0, negative = 0, zero = 0;
    mfem::real_t minimum = std::numeric_limits<mfem::real_t>::infinity();
    mfem::real_t maximum = -minimum;
    json samples = json::array();
    for (std::int64_t i = 0; i < count; ++i)
    {
      const auto value = data[i];
      if (std::isfinite(value))
      {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        negative += value < 0;
        zero += value == 0;
      }
      else
      {
        nan += std::isnan(value);
        posinf += value == std::numeric_limits<mfem::real_t>::infinity();
        neginf += value == -std::numeric_limits<mfem::real_t>::infinity();
        if (samples.size() < 8)
        {
          samples.push_back({{"index", i}, {"value", std::isnan(value) ? "nan" : (value > 0 ? "+inf" : "-inf")}});
        }
      }
    }
    const auto nonfinite = nan + posinf + neginf;
    record["summary"] = {{"nan", nan}, {"positive_infinity", posinf},
      {"negative_infinity", neginf}, {"nonfinite", nonfinite}, {"negative_finite", negative},
      {"zero", zero}, {"finite", count - nonfinite}, {"invalid_samples", samples}};
    record["summary"]["minimum"] = std::isfinite(minimum) ? json(minimum) : json(nullptr);
    record["summary"]["maximum"] = std::isfinite(maximum) ? json(maximum) : json(nullptr);
    record["snapshot"] = nullptr;
    if (impl->snapshots)
    {
      const auto bytes = static_cast<std::uint64_t>(count) * sizeof(mfem::real_t);
      Require(bytes <= impl->max_bytes - impl->bytes_written, "snapshot byte budget exceeded at " + name);
      const auto filename = "rank" + Padded(impl->rank) + ".call" + Padded(impl->call) + "." + name + ".bin";
      const auto path = impl->output / filename;
      Require(!fs::exists(path), "snapshot already exists: " + path.string());
      std::ofstream stream(path, std::ios::binary);
      Require(stream.good(), "cannot open snapshot " + path.string());
      if (bytes)
      {
        stream.write(reinterpret_cast<const char *>(data), bytes);
      }
      stream.close();
      Require(stream.good(), "cannot write snapshot " + path.string());
      impl->bytes_written += bytes;
      record["snapshot"] = filename;
    }
    impl->Append(record);
  }

  void ArrayInspector::EndCall()
  {
    if (impl->selected)
    {
      auto record = impl->context;
      record["kind"] = "end";
      record["arrays"] = impl->recorded.size();
      record["snapshot_bytes_total"] = impl->bytes_written;
      impl->Append(record);
      impl->selected = false;
    }
  }
}
